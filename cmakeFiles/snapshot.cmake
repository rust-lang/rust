CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# snapshot rustc binary
#SET(rustc "${BuildStage0Dir}/bin/rustc")
#SET(rustcCheck "${BuildStage0Dir}/bin/rustcCheck")

#IF(NOT EXISTS ${rustc})
# Get snapshot
	ADD_CUSTOM_TARGET(
		snapshot
		COMMAND ${CMAKE_COMMAND}
			-E make_directory ${BuildDlDir}
		COMMAND ${CMAKE_COMMAND}
			-E make_directory ${BuildRoot}/${HostTriple}/stage0/bin
		COMMAND ${CMAKE_COMMAND}
			-E make_directory ${BuildRoot}/${HostTriple}/stage0/lib
		COMMAND export CFG_SRC_DIR=${RustRoot} 
			&& ${RustRoot}/src/etc/get-snapshot.py ${HostTriple}
		COMMAND ${CMAKE_COMMAND}
			-E copy_directory 
			${BuildRoot}/${HostTriple}/stage0 ${BuildStageDir}/stage0
		COMMENT "Get snapshot for ${HostTriple}"
		WORKING_DIRECTORY ${BuildRoot}
		)
#ENDIF(NOT EXISTS ${rustc})

