import os
import random
import time


from enum import Enum

import numpy as np
from PIL import Image
from Settings.sj_settings import default_sim_settings, make_cfg
from Model.BaseModel import ShortestPathModel
from Agent.agent_cfg import custom_agent_config

import habitat_sim
import habitat_sim.agent
import habitat_sim.utils.datasets_download as data_downloader
from habitat_sim.nav import ShortestPath
from habitat_sim.physics import MotionType
from habitat_sim.utils.common import d3_40_colors_rgb

"""
Idea: a simple runner APT for that can take in an agent instruction and travel in the habitat simulator
"""

_barrier = None


class DemoRunnerType(Enum):
    BENCHMARK = 1
    EXAMPLE = 2
    AB_TEST = 3


class ABTestGroup(Enum):
    CONTROL = 1
    TEST = 2


class ModelControlledRunner:
    # modify to have default initiate sim setting
    def __init__(self, sim_settings=default_sim_settings, simulator_demo_type=DemoRunnerType.EXAMPLE):
        if simulator_demo_type == DemoRunnerType.EXAMPLE:
            self.set_sim_settings(sim_settings)
        self._demo_type = simulator_demo_type
        
        self.start_state = self.init_common()
        
        # shuijie defined model
        # self._defined_model = RandomActionModel()
        self._defined_model = ShortestPathModel(self.start_state.position,
                                                self._sim_settings["goal_position"],
                                                self._sim,
                                                self._sim_settings["default_agent"])
        
        # need to initialise
        self.current_observation = None
        
    ################ Data output functions #########################################################################
    def _create_output_directories(self):
        """Create directories for storing different types of observations"""
        base_dir = "./test"
        directories = [
            os.path.join(base_dir, "rgba"),
            os.path.join(base_dir, "depth"),
            os.path.join(base_dir, "sem"),
        ]

        # If in AB_TEST mode, add control and test subdirectories
        if self._demo_type == DemoRunnerType.AB_TEST:
            ab_test_dirs = []
            for dir_path in directories:
                ab_test_dirs.extend([
                    os.path.join(dir_path, "control"),
                    os.path.join(dir_path, "test")
                ])
            directories.extend(ab_test_dirs)

        # Create all required directories
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
            
    def save_color_observation(self, obs, total_frames):
        color_obs = obs["color_sensor"]
        color_img = Image.fromarray(color_obs, mode="RGBA")
        print(f"sjTest: saving image {color_img}")
        if self._demo_type == DemoRunnerType.AB_TEST:
            if self._group_id == ABTestGroup.CONTROL:
                color_img.save("./test/rgba/control/%05d.png" % total_frames)
            else:
                color_img.save("./test/rgba/test/%05d.png" % total_frames)
        else:
            color_img.save("./test/rgba/%05d.png" % total_frames)
            
    def save_semantic_observation(self, obs, total_frames):
        semantic_obs = obs["semantic_sensor"]
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        if self._demo_type == DemoRunnerType.AB_TEST:
            if self._group_id == ABTestGroup.CONTROL:
                semantic_img.save("./test/sem/control/%05d.png" % total_frames)
            else:
                semantic_img.save("./test/sem/test/%05d.png" % total_frames)
        else:
            semantic_img.save("./test/sem/%05d.png" % total_frames)

    def save_depth_observation(self, obs, total_frames):
        depth_obs = obs["depth_sensor"]
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        if self._demo_type == DemoRunnerType.AB_TEST:
            if self._group_id == ABTestGroup.CONTROL:
                depth_img.save("./test/depth/control/%05d.png" % total_frames)
            else:
                depth_img.save("./test/depth/test/%05d.png" % total_frames)
        else:
            depth_img.save("./test/depth/%05d.png" % total_frames)

    def output_semantic_mask_stats(self, obs, total_frames):
        semantic_obs = obs["semantic_sensor"]
        counts = np.bincount(semantic_obs.flatten())
        total_count = np.sum(counts)
        print(f"Pixel statistics for frame {total_frames}")
        for object_i, count in enumerate(counts):
            sem_obj = self._sim.semantic_scene.objects[object_i]
            cat = sem_obj.category.name()
            pixel_ratio = count / total_count
            if pixel_ratio > 0.01:
                print(f"obj_id:{sem_obj.id},category:{cat},pixel_ratio:{pixel_ratio}")
                
                
    ################ Agent functions #########################################################################
    
    def set_sim_settings(self, sim_settings):
        self._sim_settings = sim_settings.copy()
    
    def init_common(self):
        """
       Initialize the common components of the simulator environment.

       This function handles the core setup tasks including:
       - Configuring simulation settings for semantic sensing
       - Creating the simulator configuration
       - Downloading test scenes if not present locally
       - Initializing the Habitat simulator instance
       - Setting random seeds for reproducibility
       - Computing/recomputing navigation mesh if needed
       - Placing the agent at a random valid starting position

       Returns:
           AgentState: The initial state of the default agent after placement

       Notes:
           - Downloads test scenes automatically if the default scene file is missing
           - Seeds both Python's random and the simulator's RNG for consistency
           - Navigation mesh recomputation can be forced via sim_settings["recompute_navmesh"]
           - Uses default NavMesh settings when recomputing
       """
        self._sim_settings["semantic_sensor"] = True
        self._sim_settings["print_semantic_scene"] = True
        print(self._sim_settings)
        self._cfg = make_cfg(self._sim_settings)
        scene_file = self._sim_settings["scene"]

        if (
            not os.path.exists(scene_file)
            and scene_file == default_sim_settings["scene"]
        ):
            print(
                "Test scenes not downloaded locally, downloading and extracting now..."
            )
            data_downloader.main(["--uids", "habitat_test_scenes"])
            print("Downloaded and extracted test scenes data.")

        # create a simulator (Simulator python class object, not the backend simulator)
        self._sim = habitat_sim.Simulator(self._cfg)
        
        # self._cfg.create_renderer = True
        
        # self._sim = habitat_sim.Simulator(habitat_sim.Configuration(self._cfg, [custom_agent_config()]))

        random.seed(self._sim_settings["seed"])
        self._sim.seed(self._sim_settings["seed"])

        recompute_navmesh = self._sim_settings.get("recompute_navmesh")
        if recompute_navmesh or not self._sim.pathfinder.is_loaded:
            if not self._sim_settings["silent"]:
                print("Recomputing navmesh")
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            self._sim.recompute_navmesh(self._sim.pathfinder, navmesh_settings)
            
        # initialize the agent at a random start state
        return self.init_agent_state(self._sim_settings["default_agent"])
    
        
    def init_agent_state(self, agent_id):
        """
        Initialize an agent with a random starting position on the first floor of the environment.

        Args:
            agent_id: The identifier for the agent to be initialized

        Returns:
            start_state: The initial state of the agent, containing position and rotation information

        Notes:
            - Uses the simulator's pathfinder to get random navigable points
            - Makes up to 100 attempts to find a valid starting position on the first floor (y <= 0.5)
            - If silent mode is not enabled, prints the initial position and rotation

        Example:
            initial_state = env.init_agent_state("agent_0")
        """
        
        # initialize the agent at a random start state
        agent = self._sim.initialize_agent(agent_id)
        start_state = agent.get_state()

        # force starting position on first floor (try 100 samples)
        num_start_tries = 0
        while start_state.position[1] > 0.5 and num_start_tries < 100:
            start_state.position = self._sim.pathfinder.get_random_navigable_point()
            num_start_tries += 1
        agent.set_state(start_state)

        if not self._sim_settings["silent"]:
            print(
                "start_state.position\t",
                start_state.position,
                "start_state.rotation\t",
                start_state.rotation,
            )

        return start_state
    
    def compute_shortest_path(self, start_pos, end_pos):
        """
        Calculate the shortest path between two points using the simulator's pathfinder.

        Args:
            start_pos: The starting position coordinates
            end_pos: The target end position coordinates

        Notes:
            - Updates the internal _shortest_path object with the requested start and end positions
            - Uses the simulator's pathfinder to compute the optimal path
            - Prints the geodesic distance (shortest path length) after computation

        The geodesic distance represents the length of the shortest possible path between
        the points, taking into account navigable areas and obstacles in the environment.
        """
        self._shortest_path.requested_start = start_pos
        self._shortest_path.requested_end = end_pos
        self._sim.pathfinder.find_path(self._shortest_path)
        print("shortest_path.geodesic_distance", self._shortest_path.geodesic_distance)
        
    def get_agent_actions(self):
        """
        Retrieve the list of available actions for the default agent.

        Returns:
           list: Keys representing the possible actions in the agent's action space.

        Notes:
           - Actions are obtained from the agent's configuration in _cfg.agents
           - Uses the default_agent specified in sim_settings
        """
        return list(
            self._cfg.agents[self._sim_settings["default_agent"]].action_space.keys()
        )
        
    def do_time_steps(self):
        """
        Execute the main simulation loop, performing physics updates and observations for a specified number of frames.

        Returns:
            dict: Performance metrics containing:
                - total_time: Total wall clock time for simulation
                - frame_time: Average time per frame
                - fps: Frames per second
                - time_per_step: List of individual step execution times
                - avg_sim_step_time: Average simulation step time

        Notes:
            - If physics is enabled (removed for now):
                * Initializes a physics test scene with configurable number of objects
                * Applies random forces/translations to objects based on their motion type

            - For each time step:
                * Executes a random action from the agent's action space
                * Updates physics state if enabled
                * Captures observations (color, depth, semantic)
                * Saves sensor observations as images if configured
                * Computes shortest paths if enabled
                * Tracks performance metrics

            - Simulation continues until max_frames is reached (specified in sim_settings)

            - Performance timing excludes the first frame to avoid initialization overhead
        """
        # get the rigid object manager, which provides direct
        # access to objects
        rigid_obj_mgr = self._sim.get_rigid_object_manager()

        total_sim_step_time = 0.0
        total_frames = 0
        start_time = time.time()
        action_names = list(
            self._cfg.agents[self._sim_settings["default_agent"]].action_space.keys()
        )

        ####### REMOVE THE PHYSICS TEST FOR NOW TO MAKE THINGS EASIER
        # # load an object and position the agent for physics testing
        # if self._sim_settings["enable_physics"]:
        #     self.init_physics_test_scene(
        #         num_objects=self._sim_settings.get("num_objects")
        #     )
        #     print("active object names: " + str(rigid_obj_mgr.get_object_handles()))

        time_per_step = []

        while total_frames < self._sim_settings["max_frames"]:
            
            if total_frames == 1:
                start_time = time.time()
                            
            start_step_time = time.time()

            # apply kinematic or dynamic control to all objects based on their MotionType
            if self._sim_settings["enable_physics"]:
                obj_names = rigid_obj_mgr.get_object_handles()
                for obj_name in obj_names:
                    rand_nudge = np.random.uniform(-0.05, 0.05, 3)
                    obj = rigid_obj_mgr.get_object_by_handle(obj_name)
                    if obj.motion_type == MotionType.KINEMATIC:
                        obj.translate(rand_nudge)
                    elif obj.motion_type == MotionType.DYNAMIC:
                        obj.apply_force(rand_nudge, np.zeros(3))

            # get "interaction" time
            total_sim_step_time += time.time() - start_step_time
            
            
           
            
            # initialise the observation, for now just turn left abit
            if self.current_observation is None:       
                self.current_observation = self._sim.step("turn_left")
                
            action = self._defined_model.action(self.current_observation, action_names)
            
            observations = self._sim.step(action)
            # record the new observations
            self.current_observation = observations
 
            if not self._sim_settings["silent"]:
                print("action", action)
            
            time_per_step.append(time.time() - start_step_time)

            # get simulation step time without sensor observations
            total_sim_step_time += self._sim._previous_step_time

            # print(f"test {is_save} ######################################################")
            # if self._sim_settings["save_png"]                
            #     if self._sim_settings["color_sensor"]:
            #         self.save_color_observation(observations, total_frames)
            #     if self._sim_settings["depth_sensor"]:
            #         self.save_depth_observation(observations, total_frames)
            #     if self._sim_settings["semantic_sensor"]:
            #         self.save_semantic_observation(observations, total_frames)
            print(f"test saving image ######################################################")
            
            # no semantics for now
            self.save_semantic_observation(observations, total_frames)
            self.save_depth_observation(observations, total_frames)
            self.save_color_observation(observations, total_frames)

            total_frames += 1

        end_time = time.time()
        perf = {"total_time": end_time - start_time}
        perf["frame_time"] = perf["total_time"] / total_frames
        perf["fps"] = 1.0 / perf["frame_time"]
        perf["time_per_step"] = time_per_step
        perf["avg_sim_step_time"] = total_sim_step_time / total_frames

        return perf
    
    def run(self):
        """
       Execute the main simulation workflow from initialization through completion.

       The function performs the following sequence:
       1. Initializes common simulator components and gets agent's start state
       2. Creates output directories for data storage
       3. If enabled:
           - Computes shortest path to goal position
           - Calculates action sequence to reach goal using greedy follower
       4. Prints semantic scene information
       5. Executes main simulation loop with timesteps
       6. Cleans up simulator resources

       Returns:
           dict: Performance metrics from the simulation run including timing data
               (See do_time_steps() for detailed metrics structure)

       Notes:
           - Shortest path computation requires compute_shortest_path=True in sim_settings
           - Action path planning requires compute_action_shortest_path=True in sim_settings
           - Properly closes and deallocates simulator resources on completion
       """
        
        
        self._create_output_directories()

        # initialize and compute shortest path to goal
        if self._sim_settings["compute_shortest_path"]:
            self._shortest_path = ShortestPath()
            self.compute_shortest_path(
                self.start_state.position, self._sim_settings["goal_position"]
            )

        # set the goal headings, and compute action shortest path
        if self._sim_settings["compute_action_shortest_path"]:
            agent_id = self._sim_settings["default_agent"]
            self.greedy_follower = self._sim.make_greedy_follower(agent_id=agent_id)

            self._action_path = self.greedy_follower.find_path(
                self._sim_settings["goal_position"]
            )
            print("len(action_path)", len(self._action_path))

        perf = self.do_time_steps()
        self._sim.close()
        del self._sim

        return perf
    

def run_example():
    """
    Run simulation examples and collect performance metrics.

    Executes one or more simulation runs using DemoRunner and reports detailed 
    performance statistics. For multiple runs, calculates and reports averages.

    Performance metrics reported:
    - Frame rate (FPS)
    - Frame time (ms)
    - Per-step timing
    - When multiple runs:
       * Average FPS across runs
       * Average frame time
       * Average step time

    Notes:
       - Currently configured to run 1 iteration
       - Previously had FPS regression testing (commented out)
       - Prints detailed performance data to console
       - For multiple runs, prints both individual and averaged metrics
    """
    perfs = []
    for _i in range(1):
        demo_runner = ModelControlledRunner()
        perf = demo_runner.run()
        perfs.append(perf)

        # print(" ========================= Performance ======================== ")
        # print(
        #     " %d x %d, total time %0.2f s,"
        #     % (settings["width"], settings["height"], perf["total_time"]),
        #     "frame time %0.3f ms (%0.1f FPS)" % (perf["frame_time"] * 1000.0, perf["fps"]),
        # )
        # print(" ============================================================== ")

        # assert perf["fps"] > args.test_fps_regression, (
        #    "FPS is below regression threshold: %0.1f < %0.1f"
        #    % (perf["fps"], args.test_fps_regression)
        # )
    if len(perfs) > 1:
        avg_fps = 0
        avg_frame_time = 0
        avg_step_time = 0
        print("all perfs: " + str(perfs))
        for perf in perfs:
            print("----")
            print(perf["time_per_step"])
            avg_fps += perf["fps"]
            avg_frame_time += perf["frame_time"]
            for step_time in perf["time_per_step"]:
                avg_step_time += step_time
        avg_fps /= len(perfs)
        avg_frame_time /= len(perfs)
        avg_step_time /= len(perfs) * len(perfs[0]["time_per_step"])
        print("Average FPS: " + str(avg_fps))
        print("Average frame time: " + str(avg_frame_time))
        print("Average step time: " + str(avg_step_time))

def run_test():
    demo_runner = ModelControlledRunner()
    start_state = demo_runner.init_common()
    demo_runner._shortest_path = ShortestPath()
    shortestPath = demo_runner.compute_shortest_path(start_state.position, demo_runner._sim_settings["goal_position"])
    
    action_list = demo_runner.get_agent_actions()
    print(f"test: action_list {action_list}")

    
if __name__ == "__main__":
    run_test()
   
"""




   
   
   

    def init_physics_test_scene(self, num_objects):
        object_position = np.array(
            [-0.569043, 2.04804, 13.6156]
        )  # above the castle table

        # turn agent toward the object
        print("turning agent toward the physics!")
        agent_state = self._sim.get_agent(0).get_state()
        agent_to_obj = object_position - agent_state.position
        agent_local_forward = np.array([0, 0, -1.0])
        flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
        flat_dist_to_obj = np.linalg.norm(flat_to_obj)
        flat_to_obj /= flat_dist_to_obj
        # move the agent closer to the objects if too far (this will be projected back to floor in set)
        if flat_dist_to_obj > 3.0:
            agent_state.position = object_position - flat_to_obj * 3.0
        # unit y normal plane for rotation
        det = (
            flat_to_obj[0] * agent_local_forward[2]
            - agent_local_forward[0] * flat_to_obj[2]
        )
        turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))
        agent_state.rotation = quat_from_angle_axis(turn_angle, np.array([0, 1.0, 0]))
        # need to move the sensors too
        for sensor in agent_state.sensor_states:
            agent_state.sensor_states[sensor].rotation = agent_state.rotation
            agent_state.sensor_states[
                sensor
            ].position = agent_state.position + np.array([0, 1.5, 0])
        self._sim.get_agent(0).set_state(agent_state)

        # hard coded dimensions of maximum bounding box for all 3 default objects:
        max_union_bb_dim = np.array([0.125, 0.19, 0.26])

        # add some objects in a grid
        # get the rigid object attributes manager, which manages
        # templates used to create objects
        obj_template_mgr = self._sim.get_object_template_manager()
        # get the rigid object manager, which provides direct
        # access to objects
        rigid_obj_mgr = self._sim.get_rigid_object_manager()
        object_lib_size = obj_template_mgr.get_num_templates()
        object_init_grid_dim = (3, 1, 3)
        object_init_grid = {}
        assert (
            object_lib_size > 0
        ), "!!!No objects loaded in library, aborting object instancing example!!!"

        # clear the objects if we are re-running this initializer
        rigid_obj_mgr.remove_all_objects()

        for _obj_id in range(num_objects):
            # rand_obj_index = random.randint(0, object_lib_size - 1)
            # rand_obj_index = 0  # overwrite for specific object only
            rand_obj_index = self._sim_settings.get("test_object_index")
            if rand_obj_index < 0:  # get random object on -1
                rand_obj_index = random.randint(0, object_lib_size - 1)
            object_init_cell = (
                random.randint(-object_init_grid_dim[0], object_init_grid_dim[0]),
                random.randint(-object_init_grid_dim[1], object_init_grid_dim[1]),
                random.randint(-object_init_grid_dim[2], object_init_grid_dim[2]),
            )
            while object_init_cell in object_init_grid:
                object_init_cell = (
                    random.randint(-object_init_grid_dim[0], object_init_grid_dim[0]),
                    random.randint(-object_init_grid_dim[1], object_init_grid_dim[1]),
                    random.randint(-object_init_grid_dim[2], object_init_grid_dim[2]),
                )
            obj = rigid_obj_mgr.add_object_by_template_id(rand_obj_index)
            object_init_grid[object_init_cell] = obj.object_id
            object_offset = np.array(
                [
                    max_union_bb_dim[0] * object_init_cell[0],
                    max_union_bb_dim[1] * object_init_cell[1],
                    max_union_bb_dim[2] * object_init_cell[2],
                ]
            )
            obj.translation = object_position + object_offset
            print(
                "added object: "
                + str(obj.object_id)
                + " of type "
                + str(rand_obj_index)
                + " at: "
                + str(object_position + object_offset)
                + " | "
                + str(object_init_cell)
            )

    

    def print_semantic_scene(self):
        if self._sim_settings["print_semantic_scene"]:
        # if True:
            scene = self._sim.semantic_scene
            print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")
            for level in scene.levels:
                print(
                    f"Level id:{level.id}, center:{level.aabb.center},"
                    f" dims:{level.aabb.sizes}"
                )
                for region in level.regions:
                    print(
                        f"Region id:{region.id}, category:{region.category.name()},"
                        f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
                    )
                    for obj in region.objects:
                        print(
                            f"Object id:{obj.id}, category:{obj.category.name()},"
                            f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                        )
            input("Press Enter to continue...")

    

    def _bench_target(self, _idx=0):
        self.init_common()

        best_perf = None
        for _ in range(3):

            if _barrier is not None:
                _barrier.wait()
                if _idx == 0:
                    _barrier.reset()

            perf = self.do_time_steps()
            # The variance introduced between runs is due to the worker threads
            # being interrupted a different number of times by the kernel, not
            # due to difference in the speed of the code itself.  The most
            # accurate representation of the performance would be a run where
            # the kernel never interrupted the workers, but this isn't
            # feasible, so we just take the run with the least number of
            # interrupts (the fastest) instead.
            if best_perf is None or perf["frame_time"] < best_perf["frame_time"]:
                best_perf = perf

        self._sim.close()
        del self._sim

        return best_perf

    @staticmethod
    def _pool_init(b):
        global _barrier
        _barrier = b

    def benchmark(self, settings, group_id=ABTestGroup.CONTROL):
        self.set_sim_settings(settings)
        nprocs = settings["num_processes"]
        # set it anyway, but only be used in AB_TEST mode
        self._group_id = group_id

        barrier = multiprocessing.Barrier(nprocs)
        with multiprocessing.Pool(
            nprocs, initializer=self._pool_init, initargs=(barrier,)
        ) as pool:
            perfs = pool.map(self._bench_target, range(nprocs))

        res = {k: [] for k in perfs[0].keys()}
        for p in perfs:
            for k, v in p.items():
                res[k] += [v]

        return dict(
            frame_time=sum(res["frame_time"]),
            fps=sum(res["fps"]),
            total_time=sum(res["total_time"]) / nprocs,
            avg_sim_step_time=sum(res["avg_sim_step_time"]) / nprocs,
        )

    

"""

