# Test Cases for TSP Optimization

This directory contains test cases for validating the TSP optimization algorithms.

## File Format

Each test case is a JSON file with the following structure:
{
"name": "test_case_name",
"description": "Description of the test case",
"num_cities": 5,
"coordinates": [
{"name": "Location Name", "lat": 10.123456, "lng": 106.123456}
],
"optimal_distance_km": 45.2,
"optimal_route": ,
"notes": "Additional information about the test case"
}
## Available Test Cases

- **5_cities.json**: Small test case with 5 locations in Ho Chi Minh City
- **10_cities.json**: Medium test case with 10 delivery points
- **20_cities.json**: Large test case for performance testing

## Usage

Load test cases in your algorithms for validation:
import json

with open('data/test_cases/5_cities.json', 'r') as f:
test_case = json.load(f)

locations = test_case['coordinates']
expected_optimal = test_case['optimal_distance_km']
## Benchmarking

Use these test cases to:
- Validate algorithm correctness
- Compare GA vs PSO performance
- Test parameter sensitivity
- Benchmark runtime performance