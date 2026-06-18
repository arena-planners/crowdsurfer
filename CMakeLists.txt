cmake_minimum_required(VERSION 3.12)
project(arena_planners_crowdsurfer)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_venv REQUIRED)
find_package(ament_cmake_venv_uv REQUIRED)

uv_venv(PROJECTFILE pyproject.toml)

ament_package()
