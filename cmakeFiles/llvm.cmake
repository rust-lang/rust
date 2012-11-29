
SET(llvmSrcDir ${RustRoot}/src/llvm)
SET(llvmBuildDir ${BuildRoot}/llvm/${HostTriple})
SET(llvmBuildEnv ${HostllvmBuildEnv})

SET(llvmTargets --enable-targets=x86,x86_64,arm)
SET(llvmBuild --build=${HostTriple})
SET(llvmBuild --host=${HostTriple})
SET(llvmTarget --target=${HostTriple})
SET(llvmOpts 
    --enable-optimized
    --disable-docs
    --enable-bindings=none
    --disable-threads
    --disable-pthreads
    )
SET(llvmFlags 
    ${llvmTargets} 
    ${llvmOpts} 
    ${llvmBuild}
    ${llvmHost}
    ${llvmTarget}
    )

SET(llvmCXX)
SET(llvmCC)
SET(llvmCFLAGS)
SET(llvmCXXFLAGS)
SET(llvmLDFLAGS)

IF(${CMAKE_C_COMPILER} MATCHES clang)
    SET(llvmCXX clang++)
    SET(llvmCC clang)
ELSE()
    SET(llvmCXX g++)
    SET(llvmCC gcc)
ENDIF()

IF(NOT ${HostCpuType} MATCHES x86_64)
    SET(llvmCXX ${llvmCXX} -m32)
    SET(llvmCC ${llvmCC} -m32)
    SET(llvmCFLAGS ${llvmCFLAGS} -m32)
    SET(llvmCXXFLAGS ${llvmCXXFLAGS} -m32)
    SET(llvmLDFLAGS ${llvmLDFLAGS} -m32)
ENDIF()

ADD_CUSTOM_TARGET(
    llvm
    COMMAND
        ${CMAKE_COMMAND} -E make_directory ${llvmBuildDir}
    COMMAND
        export CXX="${llvmCXX}"
        export CC="${llvmCC}"
        export CFLAGS="${llvmCFLAGS}"
        export CXXFLAGS="${llvmCXXFLAGS}"
        export LDFLAGS="${llvmLDFLAGS}"
        && cd ${llvmBuildDir} 
        && ${llvmSrcDir}/configure ${llvmFlags}
        && make -C ${llvmBuildDir} ${llvmBuildEnv} ${BuildParallel}
    COMMENT "Building llvm"
    )
ADD_DEPENDENCIES(
	llvm
	configureSubmodules
	)
