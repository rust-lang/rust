CMAKE_MINIMUM_REQUIRED(VERSION 2.8)


# set host information

SET(HostCpuType ${CMAKE_SYSTEM_PROCESSOR})
IF(${HostCpuType} MATCHES "i386|i486|i686|i786|x86")
	SET(HostCpuType "i686")
	IF(APPLE)
		EXECUTE_PROCESS(
			COMMAND sysctl hw.optional.x86_64
			OUTPUT_VARIABLE cpucheck
			)
		IF(cpucheck MATCHES ": 1")
			SET(HostCpuType "x86_64")
		ENDIF()
	ENDIF()
ELSEIF(${HostCpuType} MATCHES "x86-64|x86_64|x64|amd64")
	SET(HostCpuType "x86_64")
ELSE()
	MESSAGE(FATAL_ERROR "Unkown CPU type")
ENDIF()

IF(APPLE)
	SET(HostOsType darwin)
	SET(HostTriple ${HostCpuType}-apple-darwin)
ELSEIF(UNIX)
	SET(HostOsType linux)
	SET(HostTriple ${HostCpuType}-unknown-linux-gnu)
ELSE()
	MESSAGE(FATAL_ERROR "Unknown OS")
ENDIF()

SET(HostSharedLibPrefix ${CMAKE_SHARED_LIBRARY_PREFIX})
SET(HostSharedLibSuffix ${CMAKE_SHARED_LIBRARY_SUFFIX})
SET(HostStaticLibPrefix ${CMAKE_STATIC_LIBRARY_PREFIX})
SET(HostStaticLibSuffix ${CMAKE_STATIC_LIBRARY_SUFFIX})
SET(HostExecSuffix ${CMAKE_EXECUTABLE_SUFFIX})

# set target information

IF(TargetOsType AND TargetCpuType)
	IF(NOT ${TargetOsType} STREQUAL "android")
		MESSAGE(FATAL_ERROR "Unknown target OS")
	ENDIF()
	IF(NOT ${TargetCpuType} STREQUAL "arm")
		MESSAGE(FATAL_ERROR "Unknown target CPU type")
	ENDIF()
	
	IF(${TargetOsType} STREQUAL "android")
		SET(TargetTriple ${TargetCpuType}-unknown-android)
		
		SET(TargetSharedLibPrefix lib)
		SET(TargetSharedLibSuffix .so)
		SET(TargetStaticLibPrefix lib)
		SET(TargetStaticLibSuffix .a)
		SET(TargetExecSuffix)
	ENDIF()
	
	SET(IsCrossCompile TRUE)
	MESSAGE(STATUS "======================================================")
	MESSAGE(STATUS "  Cross Target: ${TargetTriple}")
	MESSAGE(STATUS "======================================================")

ELSE()
	SET(IsCrossCompile FALSE)
ENDIF()



# gcc setting

# host

SET(${HostTriple}cc ${CMAKE_C_COMPILER})
SET(${HostTriple}cxx ${CMAKE_CXX_COMPILER})
SET(${HostTriple}ar ${CMAKE_AR})

IF(${HostOsType} STREQUAL linux)
	SET(HostCFlags 
		-DRUST_NDEBUG -MMD -MP -fPIC -O2 -Wall -Werror -g -fno-omit-frame-pointer)
	SET(HostLinkFlags -shared -fPIC -ldl -lpthread -lrt -g)
	IF(${HostCpuType} STREQUAL x86_64)
		SET(HostCFlags ${HostCFlags} -m64)
		SET(HostLinkFlags ${HostLinkFlags} -m64)
	ELSE()
		SET(HostCFlags ${HostCFlags} -m32)
		SET(HostLinkFlags ${HostLinkFlags} -m32)
	ENDIF()
	SET(HostCxxFlags ${HostCFlags} -fno-rtti)
	SET(HostDefFlags -Wl,--export-dynamic,--dynamic-list=)
	SET(HostPreLibFlags -Wl,-whole-archive)
	SET(HostPostLibFlags -Wl,-no-whole-archive -Wl,-znoexecstack)
	SET(HostllvmBuildEnv CXXFLAGS=-fno-omit-frame-pointer)
	SET(HostLinkPthread -lpthread)
ELSE() # darwin
	SET(HostCFlags 
		-DRUST_NDEBUG -MMD -MP -fPIC -O2 -Wall -Werror -g)
	SET(HostLinkFlags 
		-dynamiclib -lpthread -framework CoreServices -Wl,-no_compact_unwind)
	IF(${HostCpuType} STREQUAL x86_64)
		SET(HostCFlags ${HostCFlags} -m64 -arch x86_64)
		SET(HostLinkFlags ${HostLinkFlags} -m64)
	ELSE()
		SET(HostCFlags ${HostCFlags} -m32 -arch i386)
		SET(HostLinkFlags ${HostLinkFlags} -m32)
	ENDIF()
	SET(HostCxxFlags ${HostCFlags} -fno-rtti)
	SET(HostDefFlags -Wl,-exported_symbols_list,)
	SET(HostPreLibFlags)
	SET(HostPostLibFlags)
	SET(HostllvmBuildEnv)
	SET(HostLinkPthread -lpthread)
ENDIF()

# target

IF(TargetOsType STREQUAL "android")
	IF(NOT Toolchain) 
		MESSAGE(FATAL_ERROR "no toolchain information\n"
			 "use -DToolchain=<android tool chain directory>")
	ENDIF()
	IF(NOT EXISTS ${Toolchain})
		MESSAGE(FATAL_ERROR "no directory - ${Toolchain}\n"
			"use -DToolchain=<android tool chain directory>")
	ENDIF()

	SET(${TargetTriple}cc ${Toolchain}/bin/arm-linux-androideabi-gcc)
	SET(${TargetTriple}cxx ${Toolchain}/bin/arm-linux-androideabi-g++)
	SET(${TargetTriple}ar ${Toolchain}/bin/arm-linux-androideabi-ar)
	SET(${TargetTriple}include ${Toolchain}/sysroot/usr/include)
	SET(${TargetTriple}lib ${Toolchain}/sysroot/usr/lib)
	SET(${TargetTriple}sysroot ${Toolchain}/sysroot)

	IF(NOT EXISTS ${${TargetTriple}cc})
		MESSAGE(FATAL_ERROR "There is no gcc compiler in ${Toolchain}/bin")
	ENDIF()
	IF(NOT EXISTS ${${TargetTriple}cxx})
		MESSAGE(FATAL_ERROR "There is no g++ compiler in ${Toolchain}/bin")
	ENDIF()
	IF(NOT EXISTS ${${TargetTriple}ar})
		MESSAGE(FATAL_ERROR "There is no ar in ${Toolchain}/bin")
	ENDIF()

	MESSAGE(STATUS "Target gcc - ${${TargetTriple}cc}")
	MESSAGE(STATUS "Target g++ - ${${TargetTriple}cxx}")
	MESSAGE(STATUS "Target ar  - ${${TargetTriple}ar}")

	SET(TargetCFlags 
		-DRUST_NDEBUG -MMD -MP -fPIC -O2 -Wall -g -fno-omit-frame-pointer 
		-D__arm__ -DANDROID -D__ANDROID__
		-I${Toolchain}/arm-linux-androideabi/include/c++/4.6
		-I${Toolchain}/arm-linux-androideabi/include/c++/4.6/arm-linux-androideabi
		)
	SET(TargetLinkFlags -shared -fPIC -ldl -g -lm -lsupc++ -lgnustl_shared)
	SET(TargetCxxFlags ${TargetCFlags} -fno-rtti)
	SET(TargetDefFlags -Wl,--export-dynamic,--dynamic-list=)
	SET(TargetPreLibFlags -Wl,-whole-archive)
	SET(TargetPostLibFlags -Wl,-no-whole-archive -Wl,-znoexecstack)
	SET(TargetLinkPthread)
ENDIF()

##SET(CMAKE_C_COMPILER "gcc")
##SET(CMAKE_CXX_COMILER "g++")
#SET(CMAKE_C_FLAGS "${GCC_CFLAGS}")
#SET(CMAKE_CXX_FLAGS "${GCC_CXXFLAGS}")
#SET(CMAKE_C_LINK_FLAGS "${GCC_LINK_FLAGS}")
#SET(CMAKE_CXX_LINK_FLAGS "${GCC_LINK_FLAGS}")


MACRO(doMakeDef osType defName defIn defOut)
	IF(${osType} MATCHES "linux|android")
		ADD_CUSTOM_COMMAND(
			OUTPUT ${defOut}
			COMMAND ${CMAKE_COMMAND} -E copy ${defIn} ${defOut}
			COMMAND echo '{' > ${defOut} 
				&& sed 's/.$$/&\;/' ${defIn} >> ${defOut}
				&& echo '}\;' >> ${defOut}
			DEPENDS ${defIn}
			)
		ADD_CUSTOM_TARGET(
			${defName}
			DEPENDS ${defOut}
			)
	ELSEIF(${osType} STREQUAL darwin)
		ADD_CUSTOM_COMMAND(
			OUTPUT ${defOut}
			COMMAND ${CMAKE_COMMAND} -E copy ${defIn} ${defOut}
			COMMAND sed 's/^./_&/' ${defIn} > ${defOut}
			DEPENDS ${defIn}
			)
		ADD_CUSTOM_TARGET(
			${defName}
			DEPENDS ${defOut}
			)
	ENDIF()
ENDMACRO(doMakeDef)


SET(BuildParallel -j4)



#directories

SET(RustRoot "${CMAKE_SOURCE_DIR}")
MESSAGE(STATUS "Rust root: ${RustRoot}")

SET(BuildRoot ${CMAKE_BINARY_DIR})
SET(BuildDlDir ${BuildRoot}/dl)
SET(BuildStageDir ${BuildRoot}/stage)
SET(BuildRtDir ${BuildRoot}/rt)
SET(BuildRustllvmDir ${BuildRoot}/rustllvm)


SET(EnvVar 
	export CFG_BUILD_DIR=${BuildRoot}
	export CFG_VERSION=${RustVersionMajor}.${RustVersionMinor}
	export CFG_HOST_TRIPLE=${HostTriple}
	export CFG_LLVM_ROOT=
	export CFG_ENABLE_MINGW_CROSS=
	export CFG_PREFIX=/usr/home
	export CFG_LIBDIR=lib
	export CFG_SRC_DIR=${RustRoot}
	)

IF(EXISTS ${Toolchain})
	SET(EnvVar ${EnvVar} export PATH=$$PATH:${Toolchain}/bin
		)
ENDIF()

