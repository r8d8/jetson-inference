
file(GLOB detectnetConsoleSources *.cpp)
file(GLOB detectnetConsoleIncludes *.h )

cuda_add_executable(detectnet-console ${detectnetConsoleSources})

target_link_libraries(detectnet-console jetson-inference stdc++fs)

install(TARGETS detectnet-console DESTINATION bin)
