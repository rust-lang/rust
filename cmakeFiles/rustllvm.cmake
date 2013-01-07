
# llvm configurations
SET(llvmConfig 
	"${llvmBuildDir}/Release+Asserts/bin/llvm-config"
	)

EXECUTE_PROCESS(
	COMMAND ${llvmConfig} --bindir 
	OUTPUT_VARIABLE llvmBinDir
	)
EXECUTE_PROCESS(
	COMMAND ${llvmConfig} --includedir
	OUTPUT_VARIABLE llvmIncDir
	)
EXECUTE_PROCESS(
	COMMAND ${llvmConfig} --libdir
	OUTPUT_VARIABLE llvmLibDir
	)
EXECUTE_PROCESS(
	COMMAND ${llvmConfig} --libs ARM x86 ipo bitreader bitwriter linker asmparser jit mcjit interpreter
	OUTPUT_VARIABLE llvmLibs
	)
EXECUTE_PROCESS(
	COMMAND ${llvmConfig} --cxxflags
	OUTPUT_VARIABLE llvmCxxFlags
	)
EXECUTE_PROCESS(
	COMMAND ${llvmConfig} --ldflags
	OUTPUT_VARIABLE llvmLdFlags
	)
EXECUTE_PROCESS(
	COMMAND ${llvmConfig} --host-target
	OUTPUT_VARIABLE llvmHostTarget
	)
STRING(STRIP "${llvmCxxFlags}" llvmCxxFlags)
STRING(STRIP "${llvmLdFlags}" llvmLdFlags)
STRING(STRIP "${llvmLibs}" llvmLibs)
STRING(REPLACE " " ";" llvmCxxFlags "${llvmCxxFlags}")
STRING(REPLACE " " ";" llvmLdFlags "${llvmLdFlags}")
STRING(REPLACE " " ";" llvmLibs "${llvmLibs}")

#MESSAGE(STATUS "llvmCxxFlags: ${llvmCxxFlags}")
#MESSAGE(STATUS "llvmLdFlags: ${llvmLdflags}")
#MESSAGE(STATUS "llvmLibs: ${llvmLibs}")
#MESSAGE(STATUS "llvmHostTarget: ${llvmHostTarget}")

# rustllvm.def
SET(rustllvmDefIn ${RustRoot}/src/rustllvm/rustllvm.def.in)
SET(rustllvmDefOut ${BuildRustllvmDir}/rustllvm.${HostOsType}.def)

doMakeDef(${HostOsType} rustllvmDef ${rustllvmDefIn} ${rustllvmDefOut})

SET(CXX ${${HostTriple}cxx})

SET(SHP ${HostSharedLibPrefix})
SET(SHS ${HostSharedLibSuffix})
SET(rustllvmso ${BuildRustllvmDir}/${SHP}rustllvm${SHS})
SET(oRustWrapper ${BuildRustllvmDir}/RustWrapper.o)
SET(dRustWrapper ${BuildRustllvmDir}/RustWrapper.d)

# rustllvm source
SET(rustllvmSrc  
	${RustRoot}/src/rustllvm/RustWrapper.cpp
	)

ADD_CUSTOM_COMMAND(
	OUTPUT ${oRustWrapper}
	COMMAND ${CMAKE_COMMAND} -E make_directory ${BuildRustllvmDir}
	COMMAND ${CXX} ${HostCxxFlags} ${llvmCxxFlags}
		-MT ${oRustWrapper} -MF ${dRustWrapper}
		-c -o ${oRustWrapper} ${rustllvmSrc}
	)
ADD_CUSTOM_TARGET(
	rustllvmobj
	DEPENDS ${oRustWrapper}
	)

ADD_CUSTOM_COMMAND(
	OUTPUT ${rustllvmso}
	COMMAND ${CMAKE_COMMAND} -E make_directory ${BuildRustllvmDir}
	COMMAND ${CXX} ${HostLinkFlags} -o ${rustllvmso}
		${oRustWrapper}
		${HostDefFlags}${rustllvmDefOut}
		${llvmLdFlags}
		${HostPreLibFlags}
		${llvmLibs}
		${HostPostLibFlags}
		-ldl -lm
	DEPENDS ${oRustWrapper}
		${rustllvmDefOut}
	)
ADD_CUSTOM_TARGET(
	rustllvm
	DEPENDS ${rustllvmso}
	)
ADD_DEPENDENCIES(
	rustllvm 
	rustllvmDef
	rustllvmobj
	)

# copy to stage 
MACRO(doCopyRustllvm triple)
	SET(out0Dir ${BuildStageDir}/stage0/lib/rustc/${triple}/lib)
	SET(out1Dir ${BuildStageDir}/stage1/lib/rustc/${triple}/lib)
	SET(out2Dir ${BuildStageDir}/stage2/lib/rustc/${triple}/lib)

	SET(rustllvmPath ${rustllvmso})
	GET_FILENAME_COMPONENT(rustllvmName ${rustllvmPath} NAME)
	SET(${triple}rustllvmOut0 ${out0Dir}/${rustllvmName})
	SET(${triple}rustllvmOut1 ${out1Dir}/${rustllvmName})
	SET(${triple}rustllvmOut2 ${out2Dir}/${rustllvmName})

	ADD_CUSTOM_COMMAND(
		OUTPUT ${${triple}rustllvmOut0}
			${${triple}rustllvmOut1}
			${${triple}rustllvmOut2}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${out0Dir}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${out1Dir}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${out2Dir}
		COMMAND ${CMAKE_COMMAND} -E copy 
			${rustllvmPath} ${${triple}rustllvmOut0}
		COMMAND ${CMAKE_COMMAND} -E copy 
			${rustllvmPath} ${${triple}rustllvmOut1}
		COMMAND ${CMAKE_COMMAND} -E copy 
			${rustllvmPath} ${${triple}rustllvmOut2}
		DEPENDS rustllvm
		)
	ADD_CUSTOM_TARGET(
		${triple}_rustllvmCopy
		DEPENDS 
			${${triple}rustllvmOut0}
			${${triple}rustllvmOut1}
			${${triple}rustllvmOut2}
		)
ENDMACRO(doCopyRustllvm)

doCopyRustllvm(${HostTriple})
IF(${IsCrossCompile})
	doCopyRustllvm(${TargetTriple})
ENDIF()
