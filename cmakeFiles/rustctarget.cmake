# rustc source file
SET(driverCrate ${RustRoot}/src/driver/driver.rs)

# variables for rustc shared libraries
MACRO(doPrepareRustLib libname)
	SET(lib${libname}Dir ${RustRoot}/src/lib${libname})
	SET(lib${libname}Crate ${lib${libname}Dir}/${libname}.rc)
	SET(lib${libname}Out)
	FILE(GLOB_RECURSE lib${libname}Src
		${lib${libname}Dir}/*.rs
		${lib${libname}Dir}/*.rc
		)
ENDMACRO(doPrepareRustLib)

doPrepareRustLib(core)
doPrepareRustLib(std)
#doPrepareRustLib(syntax)
#doPrepareRustLib(rustc)

SET(SHP ${TargetSharedLibPrefix})
SET(SHS ${TargetSharedLibSuffix})
SET(STP ${TargetStaticLibPrefix})
SET(STS ${TargetStaticLibSuffix})
SET(EXS ${HostExecSuffix})

SET(stage0Out ${BuildStageDir}/stage0/lib/rustc/${TargetTriple})

# build a rustc shared library 
MACRO(doBuildRustLib stage libname deplib)
	SET(currStageDir ${BuildStageDir}/stage${stage})
	SET(currStageBin ${currStageDir}/bin)
	SET(currStageLib ${currStageDir}/lib)
	SET(currStageOut ${currStageLib}/rustc/${TargetTriple})
	SET(crate ${lib${libname}Crate})
	SET(srcs ${lib${libname}Src})

	SET(rustc ${currStageBin}/rustc${EXS})
	SET(rustcFlags --target=${TargetTriple})

	SET(output ${currStageOut}/lib/${SHP}${libname}${SHS})
	SET(${TargetTriple}${libname}${stage}out ${output})
	SET(depout)
	IF(NOT ${deplib} STREQUAL "")
		SET(depout ${${TargetTriple}${deplib}${stage}out}) 
	ENDIF()

	ADD_CUSTOM_COMMAND(
		OUTPUT ${output}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${currStageOut}/lib
		COMMAND ${EnvVar} 
			&& ${rustc} ${rustcFlags} -o ${output} ${crate}
			&& touch ${output}
		DEPENDS ${srcs} 
			${${HostTriple}rustrtOut}
			${${HostTriple}morestackOut}
			${${TargetTriple}rustrtOut}
			${${TargetTriple}morestackOut}
			${rustllvmOut}
			${depout}
		)
	ADD_CUSTOM_TARGET(
		${TargetTriple}_lib${libname}${stage}
		DEPENDS ${output}
		)
	ADD_DEPENDENCIES(
		${TargetTriple}_lib${libname}${stage}
		rustall
		)

ENDMACRO(doBuildRustLib)

# build stage
MACRO(doBuildStage stage)
	doBuildRustLib(${stage} core "")
	doBuildRustLib(${stage} std core)
	#doBuildRustLib(${stage} syntax std)
	#doBuildRustLib(${stage} rustc syntax)

	ADD_DEPENDENCIES(${TargetTriple}_libcore${stage}
		#${TargetTriple}_morestackCopy
		${TargetTriple}_rustrtCopy
		${TargetTriple}_rustllvmCopy
		)
	ADD_DEPENDENCIES(${TargetTriple}_libstd${stage} 
		${TargetTriple}_libcore${stage})
	#ADD_DEPENDENCIES(${TargetTriple}_libsyntax${stage} 
	#	${TargetTriple}_libstd${stage})
	#ADD_DEPENDENCIES(${TargetTriple}_librustc${stage} 
	#		${TargetTriple}_libsyntax${stage})

	ADD_CUSTOM_TARGET(
		${TargetTriple}_stageBuild${stage}
		)
	ADD_DEPENDENCIES(
		${TargetTriple}_stageBuild${stage}
		${TargetTriple}_libcore${stage}
		${TargetTriple}_libstd${stage}
		#${TargetTriple}_libsyntax${stage}
		#${TargetTriple}_librustc${stage}
		)
ENDMACRO(doBuildStage)

doBuildStage(2)


ADD_CUSTOM_TARGET(rusttarget ALL)
ADD_DEPENDENCIES(rusttarget 
	${TargetTriple}_stageBuild2
	)
