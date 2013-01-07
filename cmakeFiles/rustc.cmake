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
doPrepareRustLib(syntax)
doPrepareRustLib(rustc)

SET(SHP ${HostSharedLibPrefix})
SET(SHS ${HostSharedLibSuffix})
SET(STP ${HostStaticLibPrefix})
SET(STS ${HostStaticLibSuffix})
SET(EXS ${HostExecSuffix})


# build a rustc shared library 
MACRO(doBuildRustLib stage libname deplib)
	SET(currStageDir ${BuildStageDir}/stage${stage})
	SET(currStageBin ${currStageDir}/bin)
	SET(currStageLib ${currStageDir}/lib)
	SET(currStageOut ${currStageLib}/rustc/${HostTriple})
	SET(crate ${lib${libname}Crate})
	SET(srcs ${lib${libname}Src})

	SET(rustc ${currStageBin}/rustc${EXS})
	SET(rustcFlags --cfg stage${stage} -O --target=${HostTriple})

	SET(output ${currStageOut}/lib/${SHP}${libname}${SHS})
	SET(${HostTriple}${libname}${stage}out ${output})
	SET(depout)
	IF(NOT ${deplib} STREQUAL "")
		SET(depout ${${HostTriple}${deplib}${stage}out}) 
	ENDIF()

	ADD_CUSTOM_COMMAND(
		OUTPUT ${output}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${currStageOut}/lib
		COMMAND ${EnvVar} 
			&& ${rustc} ${rustcFlags} -o ${output} ${crate}
			&& touch ${output}
		DEPENDS ${srcs} 
			${${HostTriple}rustrtOut${stage}}
			${${HostTriple}morestackOut${stage}}
			${${HostTriple}rustllvmOut${stage}}
			${depout}
		)
	ADD_CUSTOM_TARGET(
		${HostTriple}_lib${libname}${stage}
		DEPENDS ${output}
		)

ENDMACRO(doBuildRustLib)


# build rustc
MACRO(doBuildRustc stage)
	SET(currStageDir ${BuildStageDir}/stage${stage})
	SET(currStageBin ${currStageDir}/bin)
	SET(currStageLib ${currStageDir}/lib)
	SET(currStageOut ${currStageLib}/rustc/${HostTriple})
	
	SET(stageBuildDir ${BuildStage${stage}Dir})
	SET(stageBuildOut ${stageBuildDir}/lib/rustc/${HostTriple})
	
	SET(rustc ${currStageBin}/rustc${EXS})
	SET(rustcFlags --cfg stage${stage} -O --target=${HostTriple} --cfg rustc)
	
	SET(output ${currStageOut}/bin/rustc${EXS})

	ADD_CUSTOM_COMMAND(
		OUTPUT ${output}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${currStageOut}/bin
		COMMAND ${EnvVar} 
			&& ${rustc} ${rustcFlags} -o ${output} ${driverCrate}
			&& touch ${output}
		DEPENDS ${srcs}
			${${HostTriple}core${stage}out}
			${${HostTriple}std${stage}out}
			${${HostTriple}syntax${stage}out}
			${${HostTriple}rustc${stage}out}
		)
	ADD_CUSTOM_TARGET(
		${HostTriple}_rustc${stage}		
		DEPENDS ${output}
		)
ENDMACRO(doBuildRustc)


# build stage
MACRO(doBuildStage stage)
	doBuildRustLib(${stage} core "")
	doBuildRustLib(${stage} std core)
	doBuildRustLib(${stage} syntax std)
	doBuildRustLib(${stage} rustc syntax)
	doBuildRustc(${stage})
	ADD_DEPENDENCIES(${HostTriple}_libcore${stage}
		${HostTriple}_morestackCopy
		${HostTriple}_rustrtCopy
		${HostTriple}_rustllvmCopy
		)
	ADD_DEPENDENCIES(${HostTriple}_libstd${stage} ${HostTriple}_libcore${stage})
	ADD_DEPENDENCIES(${HostTriple}_libsyntax${stage} ${HostTriple}_libstd${stage})
	ADD_DEPENDENCIES(${HostTriple}_librustc${stage} ${HostTriple}_libsyntax${stage})
	ADD_DEPENDENCIES(${HostTriple}_rustc${stage} ${HostTriple}_librustc${stage})

	ADD_CUSTOM_TARGET(
		stageBuild${stage}
		)
	ADD_DEPENDENCIES(
		stageBuild${stage}
		${HostTriple}_libcore${stage}
		${HostTriple}_libstd${stage}
		${HostTriple}_libsyntax${stage}
		${HostTriple}_librustc${stage}
		${HostTriple}_rustc${stage}
		)
	
	ADD_CUSTOM_TARGET(
		stageBuild${stage}back
		)
	ADD_DEPENDENCIES(
		stageBuild${stage}back
		${HostTriple}_libcore${stage}
		${HostTriple}_libstd${stage}
		${HostTriple}_libsyntax${stage}
		${HostTriple}_librustc${stage}
		${HostTriple}_rustc${stage}
		)
ENDMACRO(doBuildStage)

doBuildStage(0)
doBuildStage(1)
doBuildStage(2)

ADD_DEPENDENCIES(${HostTriple}_libcore0 snapshot)

# copy
MACRO(doCopyStageOut stage)
	MATH(EXPR prevStage ${stage}-1)
	MATH(EXPR nextStage ${stage}+1)
	SET(currStageDir ${BuildStageDir}/stage${stage})
	SET(currStageBin ${currStageDir}/bin)
	SET(currStageLib ${currStageDir}/lib)
	SET(currStageOut ${currStageDir}/lib/rustc/${HostTriple})
	SET(nextStageDir ${BuildStageDir}/stage${nextStage})
	SET(nextStageBin ${nextStageDir}/bin)
	SET(nextStageLib ${nextStageDir}/lib)
	SET(nextStageOut ${nextStageDir}/lib/rustc/${HostTriple})
	
	ADD_CUSTOM_COMMAND(
		OUTPUT 
			${nextStageLib}/${SHP}core${SHS}
			${nextStageLib}/${SHP}std${SHS}
			${nextStageLib}/${SHP}syntax${SHS}
			${nextStageLib}/${SHP}rustc${SHS}
			${nextStageBin}/rustc${EXS}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${nextStageBin}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${nextStageLib}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${nextStageOut}/bin
		COMMAND ${CMAKE_COMMAND} -E make_directory ${nextStageOut}/lib
		COMMAND ${CMAKE_COMMAND} -E copy
			${currStageOut}/lib/${STP}morestack${STS} ${nextStageLib}
		COMMAND ${CMAKE_COMMAND} -E copy
			${currStageOut}/lib/${SHP}rustllvm${SHS} ${nextStageLib}
		COMMAND ${CMAKE_COMMAND} -E copy
			${currStageOut}/lib/${SHP}rustrt${SHS} ${nextStageLib}
		COMMAND ${CMAKE_COMMAND} -E touch 
			${currStageOut}/lib/${SHP}core${SHS}
		COMMAND ${CMAKE_COMMAND} -E copy 
			${currStageOut}/lib/${SHP}core${SHS} ${nextStageLib}
		COMMAND ${CMAKE_COMMAND} -E copy 
			${currStageOut}/lib/${SHP}core-*${SHS} ${nextStageLib}
		COMMAND ${CMAKE_COMMAND} -E touch 
			${currStageOut}/lib/${SHP}std${SHS}
		COMMAND ${CMAKE_COMMAND} -E copy 
			${currStageOut}/lib/${SHP}std${SHS} ${nextStageLib}
		COMMAND ${CMAKE_COMMAND} -E copy 
			${currStageOut}/lib/${SHP}std-*${SHS} ${nextStageLib}
		COMMAND ${CMAKE_COMMAND} -E touch 
			${currStageOut}/lib/${SHP}syntax${SHS}
		COMMAND ${CMAKE_COMMAND} -E copy 
			${currStageOut}/lib/${SHP}syntax${SHS} ${nextStageLib}
		COMMAND ${CMAKE_COMMAND} -E copy 
			${currStageOut}/lib/${SHP}syntax-*${SHS} ${nextStageLib}
		COMMAND ${CMAKE_COMMAND} -E touch 
			${currStageOut}/lib/${SHP}rustc${SHS}
		COMMAND ${CMAKE_COMMAND} -E copy 
			${currStageOut}/lib/${SHP}rustc${SHS} ${nextStageLib}
		COMMAND ${CMAKE_COMMAND} -E copy 
			${currStageOut}/lib/${SHP}rustc-*${SHS} ${nextStageLib}
		COMMAND ${CMAKE_COMMAND} -E copy 
			${currStageOut}/bin/rustc${EXS} ${nextStageBin}
		COMMENT "copy stage${stage} to stage${nextStage}"
		)

	ADD_CUSTOM_TARGET(
		stageCopy${stage}
		DEPENDS 
			${nextStageLib}/${SHP}core${SHS}
			${nextStageLib}/${SHP}std${SHS}
			${nextStageLib}/${SHP}syntax${SHS}
			${nextStageLib}/${SHP}rustc${SHS}
			${nextStageBin}/rustc${EXS}
		)
	ADD_DEPENDENCIES(stageCopy${stage} stageBuild${stage})

	ADD_CUSTOM_TARGET(
		stageCopy${stage}back
		DEPENDS 
			${nextStageLib}/${SHP}core${SHS}
			${nextStageLib}/${SHP}std${SHS}
			${nextStageLib}/${SHP}syntax${SHS}
			${nextStageLib}/${SHP}rustc${SHS}
			${nextStageBin}/rustc${EXS}
		)
	ADD_DEPENDENCIES(stageCopy${stage}back stageBuild${stage}back)

ENDMACRO(doCopyStageOut)

doCopyStageOut(0)
doCopyStageOut(1)
doCopyStageOut(2)


ADD_CUSTOM_TARGET(stage0)
ADD_DEPENDENCIES(stage0 stageBuild0 stageCopy0)

ADD_CUSTOM_TARGET(stage1)
ADD_DEPENDENCIES(stage1 stageBuild1 stageCopy1)

ADD_CUSTOM_TARGET(stage2)
ADD_DEPENDENCIES(stage2 stageBuild2 stageCopy2)

ADD_DEPENDENCIES(stageBuild2back stageCopy1back)
ADD_DEPENDENCIES(stageBuild1back stageCopy0back)

ADD_CUSTOM_TARGET(rustall ALL)
ADD_DEPENDENCIES(rustall stageCopy2back)

