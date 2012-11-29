MESSAGE(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")

SET(installBin ${CMAKE_INSTALL_PREFIX}/bin)
SET(installLib ${CMAKE_INSTALL_PREFIX}/lib)
SET(Stage2Bin ${BuildStageDir}/stage2/bin)
SET(Stage2Lib ${BuildStageDir}/stage2/lib)
SET(Stage2OutLib ${Stage2Lib}/rustc/${HostTriple}/lib)
SET(installOutLib ${installLib}/rustc/${HostTriple}/lib)

SET(SHP ${HostSharedLibPrefix})
SET(SHS ${HostSharedLibSuffix})
SET(STP ${HostStaticLibPrefix})
SET(STS ${HostStaticLibSuffix})
SET(EXS ${HostExecSuffix})

ADD_CUSTOM_TARGET(
	hostInstall
    COMMAND ${CMAKE_COMMAND} -E make_directory ${installBin}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${installLib}
    COMMAND install -m755 
		${Stage2Bin}/rustc${EXS}
		${installBin}/rustc${EXS}
    COMMAND install -m755
		${Stage2Lib}/${SHP}rustrt${SHS}
		${installLib}/${SHP}rustrt${SHS}
    COMMAND install -m755
		${Stage2Lib}/${SHP}rustllvm${SHS}
		${installLib}/${SHP}rustllvm${SHS}
    COMMAND install -m644
		`ls -drt1 ${Stage2Lib}/${SHP}core-*${SHS} | tail -1`
        ${installLib}
    COMMAND install -m644
		`ls -drt1 ${Stage2Lib}/${SHP}std-*${SHS} | tail -1`
        ${installLib}
    COMMAND install -m644
		`ls -drt1 ${Stage2Lib}/${SHP}syntax-*${SHS} | tail -1`
        ${installLib}
    COMMAND install -m644
		`ls -drt1 ${Stage2Lib}/${SHP}rustc-*${SHS} | tail -1`
        ${installLib}

    COMMAND ${CMAKE_COMMAND} -E make_directory ${installOutLib}
    COMMAND install -m755
		${Stage2OutLib}/${SHP}rustrt${SHS}
		${installOutLib}/${SHP}rustrt${SHS}
    COMMAND install -m755
		${Stage2OutLib}/${STP}morestack${STS}
		${installOutLib}/${STP}morestack${STS}
    COMMAND install -m644
		`ls -drt1 ${Stage2OutLib}/${SHP}core-*${SHS} | tail -1`
        ${installOutLib}
    COMMAND install -m644
		`ls -drt1 ${Stage2OutLib}/${SHP}std-*${SHS} | tail -1`
        ${installOutLib}
    COMMAND install -m644
		`ls -drt1 ${Stage2OutLib}/${SHP}syntax-*${SHS} | tail -1`
        ${installOutLib}
    COMMAND install -m644
		`ls -drt1 ${Stage2OutLib}/${SHP}rustc-*${SHS} | tail -1`
        ${installOutLib}
    )
ADD_DEPENDENCIES(
	hostInstall
    rustall
    )

IF(${IsCrossCompile})
	SET(installBin ${CMAKE_INSTALL_PREFIX}/bin)
	SET(installLib ${CMAKE_INSTALL_PREFIX}/lib)
	SET(Stage2Bin ${BuildStageDir}/stage2/bin)
	SET(Stage2Lib ${BuildStageDir}/stage2/lib)
	SET(Stage2OutLib ${Stage2Lib}/rustc/${TargetTriple}/lib)
	SET(installOutLib ${installLib}/rustc/${TargetTriple}/lib)
	
	SET(SHP ${TargetSharedLibPrefix})
	SET(SHS ${TargetSharedLibSuffix})
	SET(STP ${TargetStaticLibPrefix})
	SET(STS ${TargetStaticLibSuffix})
	SET(EXS ${TargetExecSuffix})

	ADD_CUSTOM_TARGET(
		targetInstall
		COMMAND ${CMAKE_COMMAND} -E make_directory ${installOutLib}
		COMMAND install -m755
			${Stage2OutLib}/${SHP}rustrt${SHS}
			${installOutLib}/${SHP}rustrt${SHS}
		#COMMAND install -m755
		#	${Stage2OutLib}/${STP}morestack${STS}
		#	${installOutLib}/${STP}libmorestack${STS}
		COMMAND install -m644
			`ls -drt1 ${Stage2OutLib}/${SHP}core-*${SHS} | tail -1`
			${installOutLib}
		COMMAND install -m644
			`ls -drt1 ${Stage2OutLib}/${SHP}std-*${SHS} | tail -1`
			${installOutLib}
		)
	ADD_DEPENDENCIES(
		targetInstall
		rusttarget
		)
ENDIF()


IF(${IsCrossCompile})
	ADD_CUSTOM_TARGET(install)
	ADD_DEPENDENCIES(install hostInstall targetInstall)
ELSE()
	ADD_CUSTOM_TARGET(install)
	ADD_DEPENDENCIES(install hostInstall)
ENDIF()

