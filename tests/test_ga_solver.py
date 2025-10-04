"""
Unit Tests for Genetic Algorithm Solver
Testing GA implementation with DEAP 1.4.3

Author: Quang (Algorithm Specialist)
"""

import unittest
import numpy as np
import sys
import os

# Add project root and src to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

# Now import the modules
try:
    from src.ga_solver import GASolver
    from src.tsp_solver import TSPSolver
    IMPORTS_OK = True
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Project root: {project_root}")
    print(f"Src path: {src_path}")
    print(f"Available files in src: {os.listdir(src_path) if os.path.exists(src_path) else 'src not found'}")
    IMPORTS_OK = False

class TestGASolver(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Check if imports are working before running tests"""
        if not IMPORTS_OK:
            raise unittest.SkipTest("Cannot import required modules")
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test distance matrix (5 cities)
        self.distance_matrix = np.array([
            [0, 10, 15, 20, 25],
            [10, 0, 35, 25, 30],
            [15, 35, 0, 30, 20],
            [20, 25, 30, 0, 15],
            [25, 30, 20, 15, 0]
        ])
        
        # Small parameters for quick testing
        self.test_params = {
            'population_size': 20,
            'generations': 50,
            'crossover_prob': 0.8,
            'mutation_prob': 0.1,
            'tournament_size': 3,
            'elite_size': 2
        }
    
    def test_ga_solver_initialization(self):
        """Test GA solver initialization"""
        solver = GASolver(self.distance_matrix, **self.test_params)
        
        self.assertEqual(solver.num_cities, 5)
        self.assertEqual(solver.population_size, 20)
        self.assertEqual(solver.generations, 50)
        self.assertTrue(hasattr(solver, 'toolbox'))
    
    def test_distance_matrix_validation(self):
        """Test distance matrix validation"""
        # Valid matrix
        solver = GASolver(self.distance_matrix)
        self.assertIsNotNone(solver.distance_matrix)
        
        # Invalid matrix (non-square)
        invalid_matrix = np.array([[0, 1], [1, 0], [2, 3]])
        with self.assertRaises(ValueError):
            GASolver(invalid_matrix)
        
        # Invalid matrix (negative distances)
        negative_matrix = np.array([[0, -1], [-1, 0]])
        with self.assertRaises(ValueError):
            GASolver(negative_matrix)
    
    def test_individual_evaluation(self):
        """Test individual fitness evaluation"""
        solver = GASolver(self.distance_matrix, **self.test_params)
        
        # Test route [0, 1, 2, 3, 4]
        test_route = [0, 1, 2, 3, 4]
        fitness = solver.evaluate_individual(test_route)
        
        self.assertIsInstance(fitness, tuple)
        self.assertEqual(len(fitness), 1)
        self.assertGreater(fitness[0], 0)
        
        # Expected distance: 10 + 35 + 30 + 15 + 25 = 115
        expected_distance = 10 + 35 + 30 + 15 + 25
        self.assertAlmostEqual(fitness[0], expected_distance, places=6)
    
    def test_route_validation(self):
        """Test route validation function"""
        solver = GASolver(self.distance_matrix)
        
        # Valid route
        valid_route = [0, 1, 2, 3, 4]
        self.assertTrue(solver.validate_route(valid_route))
        
        # Invalid routes
        self.assertFalse(solver.validate_route([0, 1, 2]))  # Too short
        self.assertFalse(solver.validate_route([0, 1, 1, 2, 3]))  # Duplicate city
        self.assertFalse(solver.validate_route([0, 1, 2, 3, 5]))  # Invalid city index
        self.assertFalse(solver.validate_route([-1, 0, 1, 2, 3]))  # Negative index
    
    def test_ga_solve_method(self):
        """Test GA solve method"""
        solver = GASolver(self.distance_matrix, **self.test_params)
        
        # Run optimization
        best_route, best_distance = solver.solve()
        
        # Validate results
        self.assertIsInstance(best_route, list)
        self.assertIsInstance(best_distance, (int, float))
        self.assertEqual(len(best_route), 5)
        self.assertTrue(solver.validate_route(best_route))
        self.assertGreater(best_distance, 0)
        
        # Check that returned distance matches calculated distance
        calculated_distance = solver.calculate_route_distance(best_route)
        self.assertAlmostEqual(best_distance, calculated_distance, places=6)
    
    def test_ga_solve_with_stats(self):
        """Test GA solve with statistics"""
        solver = GASolver(self.distance_matrix, **self.test_params)
        
        # Run optimization with stats
        stats = solver.solve_with_stats()
        
        # Validate stats structure
        required_keys = [
            'algorithm', 'best_route', 'best_distance', 'runtime_seconds',
            'num_cities', 'num_iterations', 'convergence_data', 'success'
        ]
        
        for key in required_keys:
            self.assertIn(key, stats)
        
        # Validate stats values
        self.assertEqual(stats['algorithm'], 'GASolver')
        self.assertTrue(stats['success'])
        self.assertGreater(stats['runtime_seconds'], 0)
        self.assertEqual(stats['num_cities'], 5)
        self.assertGreater(stats['num_iterations'], 0)
    
    def test_convergence_tracking(self):
        """Test convergence data tracking"""
        solver = GASolver(self.distance_matrix, **self.test_params)
        
        # Run optimization
        solver.solve()
        
        # Check convergence data
        convergence_data = solver.get_convergence_data()
        
        self.assertIn('iterations', convergence_data)
        self.assertIn('best_distances', convergence_data)
        self.assertIn('timestamps', convergence_data)
        
        # Check data consistency
        iterations = convergence_data['iterations']
        distances = convergence_data['best_distances']
        
        self.assertEqual(len(iterations), len(distances))
        self.assertGreater(len(iterations), 0)
        
        # Check that distances are non-increasing (fitness should improve or stay same)
        for i in range(1, len(distances)):
            self.assertLessEqual(distances[i], distances[i-1])
    
    def test_deap_integration(self):
        """Test DEAP framework integration"""
        solver = GASolver(self.distance_matrix, **self.test_params)
        
        # Test DEAP components exist
        self.assertTrue(hasattr(solver, 'toolbox'))
        
        # Test population creation
        population = solver.create_population()
        self.assertEqual(len(population), solver.population_size)
        
        # Test individual creation
        for individual in population:
            self.assertEqual(len(individual), solver.num_cities)
            self.assertTrue(solver.validate_route(individual))
    
    def test_algorithm_parameters(self):
        """Test algorithm parameter retrieval"""
        solver = GASolver(self.distance_matrix, **self.test_params)
        
        params = solver.get_algorithm_params()
        
        expected_params = [
            'algorithm', 'population_size', 'generations',
            'crossover_probability', 'mutation_probability',
            'tournament_size', 'elite_size'
        ]
        
        for param in expected_params:
            self.assertIn(param, params)
        
        self.assertEqual(params['population_size'], self.test_params['population_size'])
        self.assertEqual(params['generations'], self.test_params['generations'])

class TestGAPerformance(unittest.TestCase):
    """Performance tests for GA solver"""
    
    @classmethod
    def setUpClass(cls):
        """Check if imports are working before running tests"""
        if not IMPORTS_OK:
            raise unittest.SkipTest("Cannot import required modules")
    
    def test_small_instance_performance(self):
        """Test GA performance on small TSP instance"""
        # 4-city TSP with known optimal solution
        distance_matrix = np.array([
            [0, 1, 4, 3],
            [1, 0, 2, 5],
            [4, 2, 0, 1],
            [3, 5, 1, 0]
        ])
        
        solver = GASolver(
            distance_matrix,
            population_size=50,
            generations=100,
            crossover_prob=0.8,
            mutation_prob=0.05
        )
        
        stats = solver.solve_with_stats()
        
        # Should find reasonable solution
        self.assertTrue(stats['success'])
        self.assertLess(stats['runtime_seconds'], 10)  # Should be fast
        self.assertLess(stats['best_distance'], 20)  # Should find decent solution
    
    def test_medium_instance_performance(self):
        """Test GA performance on medium TSP instance"""
        # 10-city random TSP
        np.random.seed(42)  # For reproducibility
        size = 10
        coords = np.random.rand(size, 2) * 100
        
        distance_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if i != j:
                    distance_matrix[i][j] = np.sqrt(
                        (coords[i][0] - coords[j][0])**2 + 
                        (coords[i][1] - coords[j][1])**2
                    )
        
        solver = GASolver(
            distance_matrix,
            population_size=100,
            generations=200
        )
        
        stats = solver.solve_with_stats()
        
        # Performance benchmarks
        self.assertTrue(stats['success'])
        self.assertLess(stats['runtime_seconds'], 30)  # Reasonable runtime
        self.assertGreater(stats['num_iterations'], 100)  # Should complete most generations

def run_tests():
    """Run all tests with detailed output"""
    if not IMPORTS_OK:
        print("❌ Cannot run tests: Import errors detected")
        print("Make sure you're running tests from the project root directory")
        print("Try: python -m pytest tests/test_ga_solver.py")
        return False
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestGASolver))
    test_suite.addTest(unittest.makeSuite(TestGAPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Results Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    # Check if running from correct directory
    current_dir = os.getcwd()
    if not os.path.exists(os.path.join(current_dir, 'src')):
        print("⚠️  Warning: 'src' directory not found in current working directory")
        print(f"Current directory: {current_dir}")
        print("Make sure you're running tests from the project root directory")
        print("\nTrying to find project root...")
        
        # Try to find project root
        search_dir = current_dir
        for _ in range(3):  # Search up to 3 levels up
            if os.path.exists(os.path.join(search_dir, 'src', 'ga_solver.py')):
                os.chdir(search_dir)
                print(f"✅ Found project root: {search_dir}")
                break
            search_dir = os.path.dirname(search_dir)
        else:
            print("❌ Could not find project root directory")
    
    # Run tests
    success = run_tests()
    
    # Exit with error code if tests failed
    if not success:
        sys.exit(1)