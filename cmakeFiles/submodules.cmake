
# update submodules

IF(EXISTS ${RustRoot}/.git)
	FIND_PROGRAM(Git git)
	MESSAGE(STATUS "Git: ${Git}")
	IF(${Git} MATCHES NOTFOUND)
		MESSAGE(FATAL_ERROR "require git")
	ENDIF()
	ADD_CUSTOM_TARGET(
		configureSubmodules
		COMMAND ${Git} submodule --quiet sync
		COMMAND ${Git} submodule --quiet update --init
		COMMAND ${Git} submodule --quiet foreach --recursive 
			'if test -e .gitmodules\; then git submodule sync\; fi'
		COMMAND ${Git} submodule --quiet update --init --recursive
		COMMAND ${Git} submodule status --recursive
		WORKING_DIRECTORY ${RustRoot}
		COMMENT "Updating submodules"
		)
ENDIF()

