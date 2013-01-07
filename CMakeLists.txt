## CMakeLists.txt
PROJECT(Rust)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# The version number
SET(RustVersionMajor 0)
SET(RustVersionMinor 5)

INCLUDE(cmakeFiles/config.cmake)
INCLUDE(cmakeFiles/submodules.cmake)
INCLUDE(cmakeFiles/snapshot.cmake)
INCLUDE(cmakeFiles/libuv.cmake)
INCLUDE(cmakeFiles/llvm.cmake)
INCLUDE(cmakeFiles/rustllvm.cmake)
INCLUDE(cmakeFiles/rt.cmake)
INCLUDE(cmakeFiles/rustc.cmake)
INCLUDE(cmakeFiles/install.cmake)
IF(${IsCrossCompile})
	INCLUDE(cmakeFiles/rustctarget.cmake)
ENDIF()


#SET(cmakeDir "${CMAKE_SOURCE_DIR}/cmakeFiles")
#SET(cmakeConfigure "${cmakeDir}/configuration.cmake")
#SET(cmakeSnapshot "${cmakeDir}/snapshot.cmake")
#SET(cmakeRustc "${cmakeDir}/rustc.cmake")
#SET(cmakellvm "${cmakeDir}/llvm.cmake")
#SET(cmakelibuv "${cmakeDir}/libuv.cmake")
#SET(cmakeInstall "${cmakeDir}/install.cmake")
#
#INCLUDE(${cmakeConfigure})
#INCLUDE(${cmakeSnapshot})
#INCLUDE(${cmakellvm})
#INCLUDE(${cmakelibuv})
#INCLUDE(${cmakeInstall})
#
#ADD_SUBDIRECTORY(src)
#
#INCLUDE(${cmakeRustc})

