# Print an optspec for argparse to handle cmd's options that are independent of any subcommand.
function __fish_x_global_optspecs
	string join \n v/verbose i/incremental config= build-dir= build= host= target= exclude= skip= include-default-paths rustc-error-format= on-fail= dry-run dump-bootstrap-shims stage= keep-stage= keep-stage-std= src= j/jobs= warnings= json-output color= bypass-bootstrap-lock rust-profile-generate= rust-profile-use= llvm-profile-use= llvm-profile-generate enable-bolt-settings skip-stage0-validation reproducible-artifact= set= ci= skip-std-check-if-no-download-rustc h/help
end

function __fish_x_needs_command
	# Figure out if the current invocation already has a command.
	set -l cmd (commandline -opc)
	set -e cmd[1]
	argparse -s (__fish_x_global_optspecs) -- $cmd 2>/dev/null
	or return
	if set -q argv[1]
		# Also print the command, so this can be used to figure out what it is.
		echo $argv[1]
		return 1
	end
	return 0
end

function __fish_x_using_subcommand
	set -l cmd (__fish_x_needs_command)
	test -z "$cmd"
	and return 1
	contains -- $cmd[1] $argv
end

complete -c x -n "__fish_x_needs_command" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_needs_command" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_needs_command" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_needs_command" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_needs_command" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_needs_command" -l exclude -d 'build paths to exclude' -r -F
complete -c x -n "__fish_x_needs_command" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_needs_command" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_needs_command" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_needs_command" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_needs_command" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_needs_command" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_needs_command" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_needs_command" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_needs_command" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_needs_command" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_needs_command" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_needs_command" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_needs_command" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_needs_command" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_needs_command" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_needs_command" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_needs_command" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_needs_command" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_needs_command" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_needs_command" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_needs_command" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_needs_command" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_needs_command" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_needs_command" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_needs_command" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_needs_command" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_needs_command" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_needs_command" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_needs_command" -a "build" -d 'Compile either the compiler or libraries'
complete -c x -n "__fish_x_needs_command" -a "check" -d 'Compile either the compiler or libraries, using cargo check'
complete -c x -n "__fish_x_needs_command" -a "clippy" -d 'Run Clippy (uses rustup/cargo-installed clippy binary)'
complete -c x -n "__fish_x_needs_command" -a "fix" -d 'Run cargo fix'
complete -c x -n "__fish_x_needs_command" -a "fmt" -d 'Run rustfmt'
complete -c x -n "__fish_x_needs_command" -a "doc" -d 'Build documentation'
complete -c x -n "__fish_x_needs_command" -a "test" -d 'Build and run some test suites'
complete -c x -n "__fish_x_needs_command" -a "miri" -d 'Build and run some test suites *in Miri*'
complete -c x -n "__fish_x_needs_command" -a "bench" -d 'Build and run some benchmarks'
complete -c x -n "__fish_x_needs_command" -a "clean" -d 'Clean out build directories'
complete -c x -n "__fish_x_needs_command" -a "dist" -d 'Build distribution artifacts'
complete -c x -n "__fish_x_needs_command" -a "install" -d 'Install distribution artifacts'
complete -c x -n "__fish_x_needs_command" -a "run" -d 'Run tools contained in this repository'
complete -c x -n "__fish_x_needs_command" -a "setup" -d 'Set up the environment for development'
complete -c x -n "__fish_x_needs_command" -a "suggest" -d 'Suggest a subset of tests to run, based on modified files'
complete -c x -n "__fish_x_needs_command" -a "vendor" -d 'Vendor dependencies'
complete -c x -n "__fish_x_needs_command" -a "perf" -d 'Perform profiling and benchmarking of the compiler using `rustc-perf`'
complete -c x -n "__fish_x_using_subcommand build" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand build" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand build" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand build" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand build" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand build" -l exclude -d 'build paths to exclude' -r -F
complete -c x -n "__fish_x_using_subcommand build" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand build" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand build" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand build" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_using_subcommand build" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand build" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand build" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand build" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand build" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand build" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand build" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand build" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand build" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand build" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand build" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand build" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand build" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand build" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand build" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand build" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand build" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand build" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand build" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand build" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand build" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand build" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand build" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand build" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_using_subcommand check" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand check" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand check" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand check" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand check" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand check" -l exclude -d 'build paths to exclude' -r -F
complete -c x -n "__fish_x_using_subcommand check" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand check" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand check" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand check" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_using_subcommand check" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand check" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand check" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand check" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand check" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand check" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand check" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand check" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand check" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand check" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand check" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand check" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand check" -l all-targets -d 'Check all targets'
complete -c x -n "__fish_x_using_subcommand check" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand check" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand check" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand check" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand check" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand check" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand check" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand check" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand check" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand check" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand check" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand check" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_using_subcommand clippy" -s A -d 'clippy lints to allow' -r
complete -c x -n "__fish_x_using_subcommand clippy" -s D -d 'clippy lints to deny' -r
complete -c x -n "__fish_x_using_subcommand clippy" -s W -d 'clippy lints to warn on' -r
complete -c x -n "__fish_x_using_subcommand clippy" -s F -d 'clippy lints to forbid' -r
complete -c x -n "__fish_x_using_subcommand clippy" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand clippy" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand clippy" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand clippy" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand clippy" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand clippy" -l exclude -d 'build paths to exclude' -r -F
complete -c x -n "__fish_x_using_subcommand clippy" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand clippy" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand clippy" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand clippy" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_using_subcommand clippy" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand clippy" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand clippy" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand clippy" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand clippy" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand clippy" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand clippy" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand clippy" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand clippy" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand clippy" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand clippy" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand clippy" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand clippy" -l fix
complete -c x -n "__fish_x_using_subcommand clippy" -l allow-dirty
complete -c x -n "__fish_x_using_subcommand clippy" -l allow-staged
complete -c x -n "__fish_x_using_subcommand clippy" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand clippy" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand clippy" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand clippy" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand clippy" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand clippy" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand clippy" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand clippy" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand clippy" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand clippy" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand clippy" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand clippy" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_using_subcommand fix" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand fix" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand fix" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand fix" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand fix" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand fix" -l exclude -d 'build paths to exclude' -r -F
complete -c x -n "__fish_x_using_subcommand fix" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand fix" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand fix" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand fix" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_using_subcommand fix" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand fix" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand fix" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand fix" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand fix" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand fix" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand fix" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand fix" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand fix" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand fix" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand fix" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand fix" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand fix" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand fix" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand fix" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand fix" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand fix" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand fix" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand fix" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand fix" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand fix" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand fix" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand fix" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand fix" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_using_subcommand fmt" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand fmt" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand fmt" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand fmt" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand fmt" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand fmt" -l exclude -d 'build paths to exclude' -r -F
complete -c x -n "__fish_x_using_subcommand fmt" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand fmt" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand fmt" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand fmt" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_using_subcommand fmt" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand fmt" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand fmt" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand fmt" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand fmt" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand fmt" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand fmt" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand fmt" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand fmt" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand fmt" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand fmt" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand fmt" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand fmt" -l check -d 'check formatting instead of applying'
complete -c x -n "__fish_x_using_subcommand fmt" -l all -d 'apply to all appropriate files, not just those that have been modified'
complete -c x -n "__fish_x_using_subcommand fmt" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand fmt" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand fmt" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand fmt" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand fmt" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand fmt" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand fmt" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand fmt" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand fmt" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand fmt" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand fmt" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand fmt" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_using_subcommand doc" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand doc" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand doc" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand doc" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand doc" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand doc" -l exclude -d 'build paths to exclude' -r -F
complete -c x -n "__fish_x_using_subcommand doc" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand doc" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand doc" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand doc" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_using_subcommand doc" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand doc" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand doc" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand doc" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand doc" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand doc" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand doc" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand doc" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand doc" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand doc" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand doc" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand doc" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand doc" -l open -d 'open the docs in a browser'
complete -c x -n "__fish_x_using_subcommand doc" -l json -d 'render the documentation in JSON format in addition to the usual HTML format'
complete -c x -n "__fish_x_using_subcommand doc" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand doc" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand doc" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand doc" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand doc" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand doc" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand doc" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand doc" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand doc" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand doc" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand doc" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand doc" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_using_subcommand test" -l test-args -d 'extra arguments to be passed for the test tool being used (e.g. libtest, compiletest or rustdoc)' -r
complete -c x -n "__fish_x_using_subcommand test" -l compiletest-rustc-args -d 'extra options to pass the compiler when running compiletest tests' -r
complete -c x -n "__fish_x_using_subcommand test" -l extra-checks -d 'comma-separated list of other files types to check (accepts py, py:lint, py:fmt, shell)' -r
complete -c x -n "__fish_x_using_subcommand test" -l compare-mode -d 'mode describing what file the actual ui output will be compared to' -r
complete -c x -n "__fish_x_using_subcommand test" -l pass -d 'force {check,build,run}-pass tests to this mode' -r
complete -c x -n "__fish_x_using_subcommand test" -l run -d 'whether to execute run-* tests' -r
complete -c x -n "__fish_x_using_subcommand test" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand test" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand test" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand test" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand test" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand test" -l exclude -d 'build paths to exclude' -r -F
complete -c x -n "__fish_x_using_subcommand test" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand test" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand test" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand test" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_using_subcommand test" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand test" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand test" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand test" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand test" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand test" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand test" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand test" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand test" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand test" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand test" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand test" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand test" -l no-fail-fast -d 'run all tests regardless of failure'
complete -c x -n "__fish_x_using_subcommand test" -l no-doc -d 'do not run doc tests'
complete -c x -n "__fish_x_using_subcommand test" -l doc -d 'only run doc tests'
complete -c x -n "__fish_x_using_subcommand test" -l bless -d 'whether to automatically update stderr/stdout files'
complete -c x -n "__fish_x_using_subcommand test" -l force-rerun -d 'rerun tests even if the inputs are unchanged'
complete -c x -n "__fish_x_using_subcommand test" -l only-modified -d 'only run tests that result has been changed'
complete -c x -n "__fish_x_using_subcommand test" -l rustfix-coverage -d 'enable this to generate a Rustfix coverage file, which is saved in `/<build_base>/rustfix_missing_coverage.txt`'
complete -c x -n "__fish_x_using_subcommand test" -l no-capture -d 'don\'t capture stdout/stderr of tests'
complete -c x -n "__fish_x_using_subcommand test" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand test" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand test" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand test" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand test" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand test" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand test" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand test" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand test" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand test" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand test" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand test" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_using_subcommand miri" -l test-args -d 'extra arguments to be passed for the test tool being used (e.g. libtest, compiletest or rustdoc)' -r
complete -c x -n "__fish_x_using_subcommand miri" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand miri" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand miri" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand miri" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand miri" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand miri" -l exclude -d 'build paths to exclude' -r -F
complete -c x -n "__fish_x_using_subcommand miri" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand miri" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand miri" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand miri" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_using_subcommand miri" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand miri" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand miri" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand miri" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand miri" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand miri" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand miri" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand miri" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand miri" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand miri" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand miri" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand miri" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand miri" -l no-fail-fast -d 'run all tests regardless of failure'
complete -c x -n "__fish_x_using_subcommand miri" -l no-doc -d 'do not run doc tests'
complete -c x -n "__fish_x_using_subcommand miri" -l doc -d 'only run doc tests'
complete -c x -n "__fish_x_using_subcommand miri" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand miri" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand miri" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand miri" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand miri" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand miri" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand miri" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand miri" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand miri" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand miri" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand miri" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand miri" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_using_subcommand bench" -l test-args -r
complete -c x -n "__fish_x_using_subcommand bench" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand bench" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand bench" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand bench" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand bench" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand bench" -l exclude -d 'build paths to exclude' -r -F
complete -c x -n "__fish_x_using_subcommand bench" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand bench" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand bench" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand bench" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_using_subcommand bench" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand bench" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand bench" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand bench" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand bench" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand bench" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand bench" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand bench" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand bench" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand bench" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand bench" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand bench" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand bench" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand bench" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand bench" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand bench" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand bench" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand bench" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand bench" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand bench" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand bench" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand bench" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand bench" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand bench" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_using_subcommand clean" -l stage -d 'Clean a specific stage without touching other artifacts. By default, every stage is cleaned if this option is not used' -r
complete -c x -n "__fish_x_using_subcommand clean" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand clean" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand clean" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand clean" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand clean" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand clean" -l exclude -d 'build paths to exclude' -r -F
complete -c x -n "__fish_x_using_subcommand clean" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand clean" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand clean" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand clean" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand clean" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand clean" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand clean" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand clean" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand clean" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand clean" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand clean" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand clean" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand clean" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand clean" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand clean" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand clean" -l all -d 'Clean the entire build directory (not used by default)'
complete -c x -n "__fish_x_using_subcommand clean" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand clean" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand clean" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand clean" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand clean" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand clean" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand clean" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand clean" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand clean" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand clean" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand clean" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand clean" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_using_subcommand dist" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand dist" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand dist" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand dist" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand dist" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand dist" -l exclude -d 'build paths to exclude' -r -F
complete -c x -n "__fish_x_using_subcommand dist" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand dist" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand dist" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand dist" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_using_subcommand dist" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand dist" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand dist" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand dist" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand dist" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand dist" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand dist" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand dist" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand dist" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand dist" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand dist" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand dist" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand dist" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand dist" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand dist" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand dist" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand dist" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand dist" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand dist" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand dist" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand dist" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand dist" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand dist" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand dist" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_using_subcommand install" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand install" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand install" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand install" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand install" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand install" -l exclude -d 'build paths to exclude' -r -F
complete -c x -n "__fish_x_using_subcommand install" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand install" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand install" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand install" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_using_subcommand install" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand install" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand install" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand install" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand install" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand install" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand install" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand install" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand install" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand install" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand install" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand install" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand install" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand install" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand install" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand install" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand install" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand install" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand install" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand install" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand install" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand install" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand install" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand install" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_using_subcommand run" -l args -d 'arguments for the tool' -r
complete -c x -n "__fish_x_using_subcommand run" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand run" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand run" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand run" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand run" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand run" -l exclude -d 'build paths to exclude' -r -F
complete -c x -n "__fish_x_using_subcommand run" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand run" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand run" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand run" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_using_subcommand run" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand run" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand run" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand run" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand run" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand run" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand run" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand run" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand run" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand run" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand run" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand run" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand run" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand run" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand run" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand run" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand run" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand run" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand run" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand run" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand run" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand run" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand run" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand run" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_using_subcommand setup" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand setup" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand setup" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand setup" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand setup" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand setup" -l exclude -d 'build paths to exclude' -r -F
complete -c x -n "__fish_x_using_subcommand setup" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand setup" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand setup" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand setup" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_using_subcommand setup" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand setup" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand setup" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand setup" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand setup" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand setup" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand setup" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand setup" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand setup" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand setup" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand setup" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand setup" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand setup" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand setup" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand setup" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand setup" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand setup" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand setup" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand setup" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand setup" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand setup" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand setup" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand setup" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand setup" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_using_subcommand suggest" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand suggest" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand suggest" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand suggest" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand suggest" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand suggest" -l exclude -d 'build paths to exclude' -r -F
complete -c x -n "__fish_x_using_subcommand suggest" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand suggest" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand suggest" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand suggest" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_using_subcommand suggest" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand suggest" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand suggest" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand suggest" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand suggest" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand suggest" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand suggest" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand suggest" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand suggest" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand suggest" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand suggest" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand suggest" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand suggest" -l run -d 'run suggested tests'
complete -c x -n "__fish_x_using_subcommand suggest" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand suggest" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand suggest" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand suggest" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand suggest" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand suggest" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand suggest" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand suggest" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand suggest" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand suggest" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand suggest" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand suggest" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_using_subcommand vendor" -l sync -d 'Additional `Cargo.toml` to sync and vendor' -r -F
complete -c x -n "__fish_x_using_subcommand vendor" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand vendor" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand vendor" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand vendor" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand vendor" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand vendor" -l exclude -d 'build paths to exclude' -r -F
complete -c x -n "__fish_x_using_subcommand vendor" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand vendor" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand vendor" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand vendor" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_using_subcommand vendor" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand vendor" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand vendor" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand vendor" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand vendor" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand vendor" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand vendor" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand vendor" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand vendor" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand vendor" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand vendor" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand vendor" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand vendor" -l versioned-dirs -d 'Always include version in subdir name'
complete -c x -n "__fish_x_using_subcommand vendor" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand vendor" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand vendor" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand vendor" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand vendor" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand vendor" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand vendor" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand vendor" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand vendor" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand vendor" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand vendor" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand vendor" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l exclude -d 'build paths to exclude' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -a "eprintln" -d 'Run `profile_local eprintln`. This executes the compiler on the given benchmarks and stores its stderr output'
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -a "samply" -d 'Run `profile_local samply` This executes the compiler on the given benchmarks and profiles it with `samply`. You need to install `samply`, e.g. using `cargo install samply`'
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -a "cachegrind" -d 'Run `profile_local cachegrind`. This executes the compiler on the given benchmarks under `Cachegrind`'
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -a "benchmark" -d 'Run compile benchmarks with a locally built compiler'
complete -c x -n "__fish_x_using_subcommand perf; and not __fish_seen_subcommand_from eprintln samply cachegrind benchmark compare" -a "compare" -d 'Compare the results of two previously executed benchmark runs'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l include -d 'Select the benchmarks that you want to run (separated by commas). If unspecified, all benchmarks will be executed' -r
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l exclude -d 'Select the benchmarks matching a prefix in this comma-separated list that you don\'t want to run' -r
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l scenarios -d 'Select the scenarios that should be benchmarked' -r -f -a "{Full\t'',IncrFull\t'',IncrUnchanged\t'',IncrPatched\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l profiles -d 'Select the profiles that should be benchmarked' -r -f -a "{Check\t'',Debug\t'',Doc\t'',Opt\t'',Clippy\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from eprintln" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l include -d 'Select the benchmarks that you want to run (separated by commas). If unspecified, all benchmarks will be executed' -r
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l exclude -d 'Select the benchmarks matching a prefix in this comma-separated list that you don\'t want to run' -r
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l scenarios -d 'Select the scenarios that should be benchmarked' -r -f -a "{Full\t'',IncrFull\t'',IncrUnchanged\t'',IncrPatched\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l profiles -d 'Select the profiles that should be benchmarked' -r -f -a "{Check\t'',Debug\t'',Doc\t'',Opt\t'',Clippy\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from samply" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l include -d 'Select the benchmarks that you want to run (separated by commas). If unspecified, all benchmarks will be executed' -r
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l exclude -d 'Select the benchmarks matching a prefix in this comma-separated list that you don\'t want to run' -r
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l scenarios -d 'Select the scenarios that should be benchmarked' -r -f -a "{Full\t'',IncrFull\t'',IncrUnchanged\t'',IncrPatched\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l profiles -d 'Select the profiles that should be benchmarked' -r -f -a "{Check\t'',Debug\t'',Doc\t'',Opt\t'',Clippy\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from cachegrind" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l include -d 'Select the benchmarks that you want to run (separated by commas). If unspecified, all benchmarks will be executed' -r
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l exclude -d 'Select the benchmarks matching a prefix in this comma-separated list that you don\'t want to run' -r
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l scenarios -d 'Select the scenarios that should be benchmarked' -r -f -a "{Full\t'',IncrFull\t'',IncrUnchanged\t'',IncrPatched\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l profiles -d 'Select the profiles that should be benchmarked' -r -f -a "{Check\t'',Debug\t'',Doc\t'',Opt\t'',Clippy\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from benchmark" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l config -d 'TOML configuration file for build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l build-dir -d 'Build directory, overrides `build.build-dir` in `bootstrap.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l build -d 'host target of the stage0 compiler' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l host -d 'host targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l target -d 'target targets to build' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l exclude -d 'build paths to exclude' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l skip -d 'build paths to skip' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l rustc-error-format -d 'rustc error format' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny\t'',warn\t'',default\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always\t'',never\t'',auto\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l set -d 'override options in bootstrap.toml' -r -f
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l ci -d 'Make bootstrap to behave as it\'s running on the CI environment or not' -r -f -a "{true\t'',false\t''}"
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -s i -l incremental -d 'use incremental compilation'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l dry-run -d 'dry run; don\'t build anything'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l dump-bootstrap-shims -d 'Indicates whether to dump the work done from bootstrap shims'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l json-output -d 'use message-format=json'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l bypass-bootstrap-lock -d 'Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l skip-stage0-validation -d 'Skip stage0 compiler validation'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -l skip-std-check-if-no-download-rustc -d 'Skip checking the standard library if `rust.download-rustc` isn\'t available. This is mostly for RA as building the stage1 compiler to check the library tree on each code change might be too much for some computers'
complete -c x -n "__fish_x_using_subcommand perf; and __fish_seen_subcommand_from compare" -s h -l help -d 'Print help (see more with \'--help\')'
