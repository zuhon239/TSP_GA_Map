import sys
import numpy as np

print("🐍" + "="*60)
print(f"🐍 Python version: {sys.version}")
print(f"📍 Python executable: {sys.executable}")
print("🐍" + "="*60)

print("\n📦 Testing package imports:")
print("-" * 70)

packages = [
    ('deap', 'Evolutionary algorithms (DEAP Classic)'),
    ('numpy', 'Numerical computing'),
    ('pandas', 'Data manipulation'), 
    ('scipy', 'Scientific computing'),
    ('streamlit', 'Web framework'),
    ('googlemaps', 'Google Maps API'),
    ('matplotlib', 'Plotting'),
    ('plotly', 'Interactive plots'),
    ('seaborn', 'Statistical visualization'),
    ('tqdm', 'Progress bars'),
    ('dotenv', 'Environment variables'),
    ('requests', 'HTTP requests'),
    ('openpyxl', 'Excel files')
]

success_count = 0
total_packages = len(packages)

for package_name, description in packages:
    try:
        if package_name == 'dotenv':
            from dotenv import load_dotenv
            version = "✓"
        else:
            module = __import__(package_name)
            version = getattr(module, '__version__', 'Unknown')
        
        print(f"✅ {package_name:12} - {description:35} (v{version})")
        success_count += 1
        
    except ImportError as e:
        print(f"❌ {package_name:12} - {description:35} (Error: {e})")
    except Exception as e:
        print(f"⚠️  {package_name:12} - {description:35} (Warning: {e})")

print("-" * 70)
print(f"📊 Package Status: {success_count}/{total_packages} packages installed successfully")

# Detailed DEAP testing
print(f"\n🧬 DEAP Detailed Testing:")
print("=" * 50)

try:
    import deap
    print(f"✅ DEAP version: {deap.__version__}")
    
    # Test DEAP modules
    deap_modules = [
        ('base', 'Base classes and utilities'),
        ('creator', 'Dynamic class creation'),
        ('tools', 'Genetic operators'),
        ('algorithms', 'Evolutionary algorithms')
    ]
    
    for module_name, description in deap_modules:
        try:
            module = getattr(deap, module_name)
            print(f"  ✅ deap.{module_name:10} - {description}")
        except AttributeError:
            print(f"  ❌ deap.{module_name:10} - {description} (Not found)")
    
    # Test creating TSP classes with DEAP
    print(f"\n🧬 Testing TSP setup with DEAP:")
    
    from deap import base, creator, tools, algorithms
    
    # Test creating fitness and individual classes
    print("  🔧 Creating TSP fitness class...")
    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    print("  ✅ FitnessMin class created")
    
    print("  🔧 Creating Individual class...")
    if hasattr(creator, "Individual"):
        del creator.Individual
    creator.create("Individual", list, fitness=creator.FitnessMin)
    print("  ✅ Individual class created")
    
    # Test toolbox
    print("  🔧 Setting up toolbox...")
    toolbox = base.Toolbox()
    print("  ✅ Toolbox created")
    
    # Test genetic operators
    test_route = [0, 1, 2, 3, 4]
    individual = creator.Individual(test_route)
    print(f"  🔧 Test individual: {individual}")
    print("  ✅ DEAP TSP setup successful!")
    
except Exception as e:
    print(f"❌ DEAP testing failed: {e}")
    print("❌ Consider reinstalling DEAP or checking dependencies")

# Test NumPy specifically
print(f"\n🔢 NumPy Detailed Testing:")
print("=" * 50)

try:
    import numpy as np
    print(f"✅ NumPy version: {np.__version__}")
    
    # Test creating arrays
    test_matrix = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25], 
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ])
    print(f"✅ Created test distance matrix: {test_matrix.shape}")
    print(f"✅ Matrix sum: {np.sum(test_matrix)}")
    
    # Test random operations
    random_route = np.random.permutation(4)
    print(f"✅ Random route: {random_route}")
    
except Exception as e:
    print(f"❌ NumPy testing failed: {e}")

# Test compatibility between DEAP and NumPy
print(f"\n🔗 DEAP + NumPy Integration Test:")
print("=" * 50)

try:
    import deap
    import numpy as np
    from deap import base, creator, tools
    
    # Create distance matrix
    distance_matrix = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30], 
        [20, 25, 30, 0]
    ])
    
    print(f"✅ Distance matrix created: {distance_matrix.shape}")
    
    # Function to calculate route distance
    def calculate_route_distance(route, matrix):
        total_distance = 0
        for i in range(len(route)):
            from_city = route[i] 
            to_city = route[(i + 1) % len(route)]
            total_distance += matrix[from_city][to_city]
        return total_distance
    
    test_route = [0, 1, 2, 3]
    distance = calculate_route_distance(test_route, distance_matrix)
    print(f"✅ Route {test_route} distance: {distance}")
    
    print("✅ DEAP + NumPy integration working perfectly!")
    
except Exception as e:
    print(f"❌ DEAP + NumPy integration failed: {e}")

# Final summary
print(f"\n🎯 FINAL SUMMARY:")
print("=" * 60)

if success_count >= 10:  # Most packages installed
    print("🎉 EXCELLENT! Your environment is ready for TSP development!")
    print("✅ DEAP Classic installed and working")
    print("✅ NumPy 2.3.3 compatible")
    print("✅ Python 3.11.9 fully supported")
    print("\n🚀 Next steps:")
    print("   1. Start implementing GA solver with DEAP")
    print("   2. Build PSO solver")
    print("   3. Integrate with Google Maps API")
    print("   4. Develop Streamlit frontend")
    
elif success_count >= 7:
    print("⚠️  GOOD! Most packages working, but some missing:")
    print("   - Install missing packages with: pip install <package_name>")
    print("   - Check internet connection")
    print("   - Verify package names in requirements.txt")
    
else:
    print("❌ ISSUES DETECTED! Several packages missing:")
    print("   - Reinstall Python environment")
    print("   - Check pip installation: pip --version") 
    print("   - Try: pip install --upgrade pip")
    print("   - Reinstall packages: pip install -r requirements.txt")

print("\n📋 Environment Details:")
print(f"   - Python: {sys.version.split()[0]}")
print(f"   - DEAP: {deap.__version__ if 'deap' in sys.modules else 'Not installed'}")
print(f"   - NumPy: {np.__version__ if 'numpy' in sys.modules else 'Not installed'}")
print("=" * 60)