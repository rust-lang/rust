SET(libuvSrcDir ${RustRoot}/src/libuv)

MACRO(doBuildLibuv triple os cpu)

	SET(${triple}libuvBuildDir ${BuildRoot}/libuv/${triple})

	IF(${cpu} STREQUAL x86_64)
		SET(${triple}libuvArch x86_64)
		SET(${triple}libuvFlags -m64 -fPIC)
	ELSEIF(${cpu} STREQUAL i686)
		SET(${triple}libuvArch ia32)
		SET(${triple}libuvFlags -m32 -fPIC)
	ELSEIF(${cpu} MATCHES "arm")
		SET(${triple}libuvArch arm)
		SET(${triple}libuvFlags -fPIC)
	ENDIF()

	IF(${os} STREQUAL darwin)
		SET(${triple}libuvOsType mac)
		SET(${triple}libuvLib 
			${${triple}libuvBuildDir}/Release/libuv.a
			)
	ELSEIF(${os} STREQUAL linux)
		SET(${triple}libuvOsType unix/linux)
		SET(${triple}libuvLib 
			${${triple}libuvBuildDir}/Release/obj.target/src/libuv/libuv.a
			)
	ELSEIF(${os} STREQUAL android)
		SET(${triple}libuvOsType unix/android)
		SET(${triple}libuvLib 
			${${triple}libuvBuildDir}/Release/obj.target/src/libuv/libuv.a
			)
		SET(${triple}libuvFlags ${${triple}libuvFlags} -DANDROID -std=gnu99)
	ENDIF()


	ADD_CUSTOM_COMMAND(
		OUTPUT ${${triple}libuvLib}
		COMMAND
			${CMAKE_COMMAND} -E make_directory ${${triple}libuvBuildDir}
		COMMAND
			make -C 
			${RustRoot}/mk/libuv/${${triple}libuvArch}/${${triple}libuvOsType}
			${BuildParallel}
			CFLAGS="${${triple}libuvFlags}"
			LDFLAGS="${${triple}libuvFlags}"
			CC="${${triple}cc}"
			CXX="${${triple}cxx}"
			AR="${${triple}ar}"
			BUILDTYPE=Release
			builddir_name="${${triple}libuvBuildDir}"
			FLOCK=uv
		)
	ADD_CUSTOM_TARGET(
		${triple}_libuv
		DEPENDS ${${triple}libuvLib}
		)
	ADD_DEPENDENCIES(
		${triple}_libuv
		configureSubmodules
		)

	MESSAGE(STATUS "libuv Target: ${triple}libuv")
ENDMACRO(doBuildLibuv)

ADD_CUSTOM_TARGET(libuv)
ADD_DEPENDENCIES(libuv ${HostTriple}_libuv)


doBuildLibuv(${HostTriple} ${HostOsType} ${HostCpuType})

IF(IsCrossCompile) 
	doBuildLibuv(${TargetTriple} ${TargetOsType} ${TargetCpuType})
ENDIF()

