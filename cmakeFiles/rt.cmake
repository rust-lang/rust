cmake_minimum_required(VERSION 2.8)


SET(rtDir "${RustRoot}/src/rt")
SET(uvDir "${RustRoot}/src/libuv")


# llvm-mc
SET(llvm_mc "${llvmBuildDir}/Release+Asserts/bin/llvm-mc")
MESSAGE(STATUS "llvm-mc: ${llvm_mc}")


# build rustrt
MACRO(doBuildRustrt type)

	MESSAGE(STATUS "rustrt build script for ${type}")

	SET(osType ${${type}OsType})
	SET(cpuType ${${type}CpuType})
	SET(triple ${${type}Triple})
	
	SET(SHP ${${type}SharedLibPrefix})
	SET(SHS ${${type}SharedLibSuffix})
	SET(STP ${${type}StaticLibPrefix})
	SET(STS ${${type}StaticLibSuffix})
	SET(EXS ${${type}ExecSuffix})

	SET(CC ${${triple}cc})
	SET(CXX ${${triple}cxx})
	SET(AR ${${triple}ar})
	SET(CFlags ${${type}CFlags})
	SET(CxxFlags ${${type}CxxFlags})
	SET(LinkFlags ${${type}LinkFlags})
	SET(DefFlags ${${type}DefFlags})
	SET(PreLibFlags ${${type}PreLibFlags})
	SET(PostLibFlags ${${type}PostLibFlags})
	SET(LinkPthread ${${type}LinkPthread})

	SET(archType ${cpuType})
	IF(${archType} STREQUAL "i686")
		SET(archType i386)
	ENDIF()

	MESSAGE(STATUS "rurstrt arch: ${archType}")

	# rustrt.def
	SET(rustrtDefIn ${RustRoot}/src/rt/rustrt.def.in)
	SET(rustrtDefOut ${BuildRtDir}/${triple}/rustrt.${osType}.def)
	doMakeDef(${osType} ${triple}_rustrtDef ${rustrtDefIn} ${rustrtDefOut})

	# source targets
	SET(${triple}rtTargets "")

	# source objects
	SET(${triple}rtObjects "")
	
	# morestack used?
	IF(${cpuType} STREQUAL arm)
		SET(useMorestack false)
	ELSE()
		SET(useMorestack true)
	ENDIF()

	# assembly source files
	IF(${useMorestack}) 
		SET(rtSS
			arch/${archType}/morestack.S
			arch/${archType}/_context.S
			arch/${archType}/ccall.S
			arch/${archType}/record_sp.S
			)
	ELSE()
		SET(rtSS
			arch/${archType}/_context.S
			arch/${archType}/ccall.S
			arch/${archType}/record_sp.S
			)
	ENDIF()

	# compile assembly
	FOREACH(s ${rtSS})
		GET_FILENAME_COMPONENT(spath ${s} PATH)
		GET_FILENAME_COMPONENT(sname ${s} NAME_WE)
		
		SET(ss ${rtDir}/${s})
		SET(bd ${BuildRtDir}/${triple}/${spath})
		SET(so ${bd}/${sname}.o)
		SET(sd ${bd}/${sname}.d)
		SET(sa ${bd}/${STP}${sname}${STS})
		IF(${cpuType} STREQUAL arm)
			ADD_CUSTOM_COMMAND(
				OUTPUT ${so}
				COMMAND ${CMAKE_COMMAND} -E make_directory ${bd}
				COMMAND ${CC} -MMD -MP -MT ${so} -MF ${sd} ${ss} -c -o ${so} 
				COMMAND ar rcs ${sa} ${so}
				DEPENDS ${ss}
				COMMENT "Building ${triple} ${sname}.S"
				)
			ADD_CUSTOM_TARGET(
				${triple}_${sname}
				DEPENDS ${so}
				)
		ELSE()
			ADD_CUSTOM_COMMAND(
				OUTPUT ${so}
				COMMAND ${CMAKE_COMMAND} -E make_directory ${bd}
				COMMAND ${CC} -E -MMD -MP -MT ${so} -MF ${sd} ${ss}
					| ${llvm_mc} -assemble -filetype=obj -triple=${triple} -o=${so} 
				COMMAND ar rcs ${sa} ${so}
				DEPENDS ${ss}
				COMMENT "Building ${triple} ${sname}.S"
				)
			ADD_CUSTOM_TARGET(
				${triple}_${sname}
				DEPENDS ${so}
				)
		ENDIF()
		
		SET(${triple}rtTargets ${${triple}rtTargets} ${${triple}_${sname}})
		IF(NOT ${sname} MATCHES morestack)
			SET(${triple}rtObjects ${${triple}rtObjects} ${so})
		ELSE()
			SET(${triple}morestacko ${so})
			SET(${triple}morestacka ${sa})
		ENDIF()
	ENDFOREACH(s)


	# add include directories
	SET(includeDir
		-I${rtDir}
		-I${rtDir}/isaac
		-I${rtDir}/uthash
		-I${rtDir}/arch/${archType}
		-I${RustRoot}/src/libuv/include
		)

	# cpp source files
	SET(rtCCxxSrc
		sync/timer.cpp
		sync/lock_and_signal.cpp
		sync/rust_thread.cpp
		rust.cpp
		rust_builtin.cpp
		rust_run_program.cpp
		rust_env.cpp
		rust_sched_loop.cpp
		rust_sched_launcher.cpp
		rust_sched_driver.cpp
		rust_scheduler.cpp
		rust_sched_reaper.cpp
		rust_task.cpp
		rust_stack.cpp
		rust_port.cpp
		rust_upcall.cpp
		rust_uv.cpp
		rust_crate_map.cpp
		rust_log.cpp
		rust_gc_metadata.cpp
		rust_port_selector.cpp
		rust_util.cpp
		circular_buffer.cpp
		isaac/randport.cpp
		miniz.cpp
		rust_kernel.cpp
		rust_abi.cpp
		rust_debug.cpp
		memory_region.cpp
		boxed_region.cpp
		arch/${archType}/context.cpp
		arch/${archType}/gpr.cpp
		linenoise/linenoise.c
		linenoise/utf8.c
		rust_android_dummy.cpp
		)

	FOREACH(src ${rtCCxxSrc})
		GET_FILENAME_COMPONENT(cext ${src} EXT)
		GET_FILENAME_COMPONENT(cpath ${src} PATH)
		GET_FILENAME_COMPONENT(cname ${src} NAME_WE)
		
		SET(bd ${BuildRtDir}/${triple}/${cpath})
		SET(cs ${rtDir}/${src})
		SET(co ${bd}/${cname}.o)
		SET(cd ${bd}/${cname}.d)
		IF(${cext} STREQUAL ".c")
			SET(_CC ${CC})
			SET(_Flags ${CFlags})
		ELSE()
			SET(_CC ${CXX})
			SET(_Flags ${CxxFlags})
		ENDIF()
		
		ADD_CUSTOM_COMMAND(
			OUTPUT ${co}
			COMMAND ${CMAKE_COMMAND} -E make_directory ${bd}
			COMMAND ${_CC} ${_Flags} -MT ${co} -MF ${cd} 
				-c -o ${co} ${includeDir} ${cs} 
			DEPENDS ${cs}
			COMMENT "Building ${triple} ${cname}${cext}"
			)
		ADD_CUSTOM_TARGET(
			${triple}_${cname}
			DEPENDS ${co}
			)
		
		SET(${triple}rtTargets ${${triple}rtTargets} ${triple}_${cname})
		SET(${triple}rtObjects ${${triple}rtObjects} ${co})
	ENDFOREACH(src)

	SET(bd ${BuildRtDir}/${triple})
	SET(${triple}rustrtso ${BuildRtDir}/${triple}/${SHP}rustrt${SHS})
	ADD_CUSTOM_COMMAND(
		OUTPUT ${${triple}rustrtso}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${bd}
		COMMAND ${CXX} ${LinkFlags} -o ${${triple}rustrtso}
			${DefFlags}${rustrtDefOut} 
			${${triple}rtObjects}
			${PostLibFlags}
			${${triple}libuvLib}
			${LinkPthread}
		DEPENDS ${${triple}rtObjects}
			${rustrtDefOut}
			${${triple}libuvLib}
		)
	ADD_CUSTOM_TARGET(
		${triple}_rustrt
		DEPENDS ${${triple}rustrtso} ${${triple}libuvLib}
		)
	ADD_DEPENDENCIES(
		${triple}_rustrt
		${${triple}rtTargets}
		${triple}_libuv
		)


	# copy librustrt to stage0 output
	SET(out0Dir ${BuildStageDir}/stage0/lib/rustc/${triple}/lib)
	SET(out1Dir ${BuildStageDir}/stage1/lib/rustc/${triple}/lib)
	SET(out2Dir ${BuildStageDir}/stage2/lib/rustc/${triple}/lib)
	SET(${triple}rustrtOut0 ${out0Dir}/${SHP}rustrt${SHS})
	SET(${triple}rustrtOut1 ${out1Dir}/${SHP}rustrt${SHS})
	SET(${triple}rustrtOut2 ${out2Dir}/${SHP}rustrt${SHS})
	MESSAGE(STATUS "ok: ${${triple}rustrtOut0}")

	ADD_CUSTOM_COMMAND(
		OUTPUT ${${triple}rustrtOut0}
			${${triple}rustrtOut1}
			${${triple}rustrtOut2}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${out0Dir}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${out1Dir}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${out2Dir}
		COMMAND ${CMAKE_COMMAND} -E copy 
			${${triple}rustrtso} ${${triple}rustrtOut0}
		COMMAND ${CMAKE_COMMAND} -E copy 
			${${triple}rustrtso} ${${triple}rustrtOut1}
		COMMAND ${CMAKE_COMMAND} -E copy 
			${${triple}rustrtso} ${${triple}rustrtOut2}
		DEPENDS ${${triple}rustrtso}
		)
	ADD_CUSTOM_TARGET(
		${triple}_rustrtCopy
		DEPENDS 
			${${triple}rustrtOut0}
			${${triple}rustrtOut1}
			${${triple}rustrtOut2}
		)
	ADD_DEPENDENCIES(
		${triple}_rustrtCopy
		${triple}_rustrt
		)

	IF(${useMorestack})
		# copy libmorestack to stage0 output
		SET(ss ${rtDir}/${s})
		SET(bd ${BuildRtDir}/${triple}/${spath})
		SET(so ${bd}/${sname}.o)
		SET(sd ${bd}/${sname}.d)
		SET(sa ${bd}/${STP}${sname}${STS})

		SET(${triple}morestackOut0 ${out0Dir}/${STP}morestack${STS})
		SET(${triple}morestackOut1 ${out1Dir}/${STP}morestack${STS})
		SET(${triple}morestackOut2 ${out2Dir}/${STP}morestack${STS})

		ADD_CUSTOM_COMMAND(
			OUTPUT ${${triple}morestackOut0} 
				${${triple}morestackOut1}
				${${triple}morestackOut2}
			COMMAND ${CMAKE_COMMAND} -E make_directory ${out0Dir}
			COMMAND ${CMAKE_COMMAND} -E make_directory ${out1Dir}
			COMMAND ${CMAKE_COMMAND} -E make_directory ${out2Dir}
			COMMAND ${CMAKE_COMMAND} -E copy 
				${${triple}morestacka} ${${triple}morestackOut0}
			COMMAND ${CMAKE_COMMAND} -E copy 
				${${triple}morestacka} ${${triple}morestackOut1}
			COMMAND ${CMAKE_COMMAND} -E copy 
				${${triple}morestacka} ${${triple}morestackOut2}
			DEPENDS ${${triple}morestacko}
			)
		ADD_CUSTOM_TARGET(
			${triple}_morestackCopy
			DEPENDS 
				${${triple}morestackOut0}
				${${triple}morestackOut1}
				${${triple}morestackOut2}
			)
	ENDIF()

ENDMACRO(doBuildRustrt)

doBuildRustrt(Host)
IF(${IsCrossCompile})
	doBuildRustrt(Target)
ENDIF()

