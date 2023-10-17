complete -c x.py -n "__fish_use_subcommand" -l config -d 'TOML configuration file for build' -r -F
complete -c x.py -n "__fish_use_subcommand" -l build-dir -d 'Build directory, overrides `build.build-dir` in `config.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_use_subcommand" -l build -d 'build target of the stage0 compiler' -r -f
complete -c x.py -n "__fish_use_subcommand" -l host -d 'host targets to build' -r -f
complete -c x.py -n "__fish_use_subcommand" -l target -d 'target targets to build' -r -f
complete -c x.py -n "__fish_use_subcommand" -l exclude -d 'build paths to exclude' -r -F
complete -c x.py -n "__fish_use_subcommand" -l skip -d 'build paths to skip' -r -F
complete -c x.py -n "__fish_use_subcommand" -l rustc-error-format -r -f
complete -c x.py -n "__fish_use_subcommand" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x.py -n "__fish_use_subcommand" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x.py -n "__fish_use_subcommand" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_use_subcommand" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_use_subcommand" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_use_subcommand" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x.py -n "__fish_use_subcommand" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny	'',warn	'',default	''}"
complete -c x.py -n "__fish_use_subcommand" -l error-format -d 'rustc error format' -r -f
complete -c x.py -n "__fish_use_subcommand" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always	'',never	'',auto	''}"
complete -c x.py -n "__fish_use_subcommand" -l llvm-skip-rebuild -d 'whether rebuilding llvm should be skipped, overriding `skip-rebuld` in config.toml' -r -f -a "{true	'',false	''}"
complete -c x.py -n "__fish_use_subcommand" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x.py -n "__fish_use_subcommand" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x.py -n "__fish_use_subcommand" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x.py -n "__fish_use_subcommand" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x.py -n "__fish_use_subcommand" -l set -d 'override options in config.toml' -r -f
complete -c x.py -n "__fish_use_subcommand" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x.py -n "__fish_use_subcommand" -s i -l incremental -d 'use incremental compilation'
complete -c x.py -n "__fish_use_subcommand" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x.py -n "__fish_use_subcommand" -l dry-run -d 'dry run; don\'t build anything'
complete -c x.py -n "__fish_use_subcommand" -l json-output -d 'use message-format=json'
complete -c x.py -n "__fish_use_subcommand" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x.py -n "__fish_use_subcommand" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x.py -n "__fish_use_subcommand" -s h -l help -d 'Print help'
complete -c x.py -n "__fish_use_subcommand" -f -a "build" -d 'Compile either the compiler or libraries'
complete -c x.py -n "__fish_use_subcommand" -f -a "check" -d 'Compile either the compiler or libraries, using cargo check'
complete -c x.py -n "__fish_use_subcommand" -f -a "clippy" -d 'Run Clippy (uses rustup/cargo-installed clippy binary)'
complete -c x.py -n "__fish_use_subcommand" -f -a "fix" -d 'Run cargo fix'
complete -c x.py -n "__fish_use_subcommand" -f -a "fmt" -d 'Run rustfmt'
complete -c x.py -n "__fish_use_subcommand" -f -a "doc" -d 'Build documentation'
complete -c x.py -n "__fish_use_subcommand" -f -a "test" -d 'Build and run some test suites'
complete -c x.py -n "__fish_use_subcommand" -f -a "bench" -d 'Build and run some benchmarks'
complete -c x.py -n "__fish_use_subcommand" -f -a "clean" -d 'Clean out build directories'
complete -c x.py -n "__fish_use_subcommand" -f -a "dist" -d 'Build distribution artifacts'
complete -c x.py -n "__fish_use_subcommand" -f -a "install" -d 'Install distribution artifacts'
complete -c x.py -n "__fish_use_subcommand" -f -a "run" -d 'Run tools contained in this repository'
complete -c x.py -n "__fish_use_subcommand" -f -a "setup" -d 'Set up the environment for development'
complete -c x.py -n "__fish_use_subcommand" -f -a "suggest" -d 'Suggest a subset of tests to run, based on modified files'
complete -c x.py -n "__fish_seen_subcommand_from build" -l config -d 'TOML configuration file for build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from build" -l build-dir -d 'Build directory, overrides `build.build-dir` in `config.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from build" -l build -d 'build target of the stage0 compiler' -r -f
complete -c x.py -n "__fish_seen_subcommand_from build" -l host -d 'host targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from build" -l target -d 'target targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from build" -l exclude -d 'build paths to exclude' -r -F
complete -c x.py -n "__fish_seen_subcommand_from build" -l skip -d 'build paths to skip' -r -F
complete -c x.py -n "__fish_seen_subcommand_from build" -l rustc-error-format -r -f
complete -c x.py -n "__fish_seen_subcommand_from build" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x.py -n "__fish_seen_subcommand_from build" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from build" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from build" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from build" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from build" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x.py -n "__fish_seen_subcommand_from build" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny	'',warn	'',default	''}"
complete -c x.py -n "__fish_seen_subcommand_from build" -l error-format -d 'rustc error format' -r -f
complete -c x.py -n "__fish_seen_subcommand_from build" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always	'',never	'',auto	''}"
complete -c x.py -n "__fish_seen_subcommand_from build" -l llvm-skip-rebuild -d 'whether rebuilding llvm should be skipped, overriding `skip-rebuld` in config.toml' -r -f -a "{true	'',false	''}"
complete -c x.py -n "__fish_seen_subcommand_from build" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from build" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from build" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from build" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x.py -n "__fish_seen_subcommand_from build" -l set -d 'override options in config.toml' -r -f
complete -c x.py -n "__fish_seen_subcommand_from build" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x.py -n "__fish_seen_subcommand_from build" -s i -l incremental -d 'use incremental compilation'
complete -c x.py -n "__fish_seen_subcommand_from build" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x.py -n "__fish_seen_subcommand_from build" -l dry-run -d 'dry run; don\'t build anything'
complete -c x.py -n "__fish_seen_subcommand_from build" -l json-output -d 'use message-format=json'
complete -c x.py -n "__fish_seen_subcommand_from build" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x.py -n "__fish_seen_subcommand_from build" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x.py -n "__fish_seen_subcommand_from build" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x.py -n "__fish_seen_subcommand_from check" -l config -d 'TOML configuration file for build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from check" -l build-dir -d 'Build directory, overrides `build.build-dir` in `config.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from check" -l build -d 'build target of the stage0 compiler' -r -f
complete -c x.py -n "__fish_seen_subcommand_from check" -l host -d 'host targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from check" -l target -d 'target targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from check" -l exclude -d 'build paths to exclude' -r -F
complete -c x.py -n "__fish_seen_subcommand_from check" -l skip -d 'build paths to skip' -r -F
complete -c x.py -n "__fish_seen_subcommand_from check" -l rustc-error-format -r -f
complete -c x.py -n "__fish_seen_subcommand_from check" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x.py -n "__fish_seen_subcommand_from check" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from check" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from check" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from check" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from check" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x.py -n "__fish_seen_subcommand_from check" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny	'',warn	'',default	''}"
complete -c x.py -n "__fish_seen_subcommand_from check" -l error-format -d 'rustc error format' -r -f
complete -c x.py -n "__fish_seen_subcommand_from check" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always	'',never	'',auto	''}"
complete -c x.py -n "__fish_seen_subcommand_from check" -l llvm-skip-rebuild -d 'whether rebuilding llvm should be skipped, overriding `skip-rebuld` in config.toml' -r -f -a "{true	'',false	''}"
complete -c x.py -n "__fish_seen_subcommand_from check" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from check" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from check" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from check" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x.py -n "__fish_seen_subcommand_from check" -l set -d 'override options in config.toml' -r -f
complete -c x.py -n "__fish_seen_subcommand_from check" -l all-targets -d 'Check all targets'
complete -c x.py -n "__fish_seen_subcommand_from check" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x.py -n "__fish_seen_subcommand_from check" -s i -l incremental -d 'use incremental compilation'
complete -c x.py -n "__fish_seen_subcommand_from check" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x.py -n "__fish_seen_subcommand_from check" -l dry-run -d 'dry run; don\'t build anything'
complete -c x.py -n "__fish_seen_subcommand_from check" -l json-output -d 'use message-format=json'
complete -c x.py -n "__fish_seen_subcommand_from check" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x.py -n "__fish_seen_subcommand_from check" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x.py -n "__fish_seen_subcommand_from check" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x.py -n "__fish_seen_subcommand_from clippy" -s A -d 'clippy lints to allow' -r
complete -c x.py -n "__fish_seen_subcommand_from clippy" -s D -d 'clippy lints to deny' -r
complete -c x.py -n "__fish_seen_subcommand_from clippy" -s W -d 'clippy lints to warn on' -r
complete -c x.py -n "__fish_seen_subcommand_from clippy" -s F -d 'clippy lints to forbid' -r
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l config -d 'TOML configuration file for build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l build-dir -d 'Build directory, overrides `build.build-dir` in `config.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l build -d 'build target of the stage0 compiler' -r -f
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l host -d 'host targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l target -d 'target targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l exclude -d 'build paths to exclude' -r -F
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l skip -d 'build paths to skip' -r -F
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l rustc-error-format -r -f
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from clippy" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny	'',warn	'',default	''}"
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l error-format -d 'rustc error format' -r -f
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always	'',never	'',auto	''}"
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l llvm-skip-rebuild -d 'whether rebuilding llvm should be skipped, overriding `skip-rebuld` in config.toml' -r -f -a "{true	'',false	''}"
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l set -d 'override options in config.toml' -r -f
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l fix
complete -c x.py -n "__fish_seen_subcommand_from clippy" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x.py -n "__fish_seen_subcommand_from clippy" -s i -l incremental -d 'use incremental compilation'
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l dry-run -d 'dry run; don\'t build anything'
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l json-output -d 'use message-format=json'
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x.py -n "__fish_seen_subcommand_from clippy" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x.py -n "__fish_seen_subcommand_from clippy" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x.py -n "__fish_seen_subcommand_from fix" -l config -d 'TOML configuration file for build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from fix" -l build-dir -d 'Build directory, overrides `build.build-dir` in `config.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from fix" -l build -d 'build target of the stage0 compiler' -r -f
complete -c x.py -n "__fish_seen_subcommand_from fix" -l host -d 'host targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from fix" -l target -d 'target targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from fix" -l exclude -d 'build paths to exclude' -r -F
complete -c x.py -n "__fish_seen_subcommand_from fix" -l skip -d 'build paths to skip' -r -F
complete -c x.py -n "__fish_seen_subcommand_from fix" -l rustc-error-format -r -f
complete -c x.py -n "__fish_seen_subcommand_from fix" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x.py -n "__fish_seen_subcommand_from fix" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from fix" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from fix" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from fix" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from fix" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x.py -n "__fish_seen_subcommand_from fix" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny	'',warn	'',default	''}"
complete -c x.py -n "__fish_seen_subcommand_from fix" -l error-format -d 'rustc error format' -r -f
complete -c x.py -n "__fish_seen_subcommand_from fix" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always	'',never	'',auto	''}"
complete -c x.py -n "__fish_seen_subcommand_from fix" -l llvm-skip-rebuild -d 'whether rebuilding llvm should be skipped, overriding `skip-rebuld` in config.toml' -r -f -a "{true	'',false	''}"
complete -c x.py -n "__fish_seen_subcommand_from fix" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from fix" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from fix" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from fix" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x.py -n "__fish_seen_subcommand_from fix" -l set -d 'override options in config.toml' -r -f
complete -c x.py -n "__fish_seen_subcommand_from fix" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x.py -n "__fish_seen_subcommand_from fix" -s i -l incremental -d 'use incremental compilation'
complete -c x.py -n "__fish_seen_subcommand_from fix" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x.py -n "__fish_seen_subcommand_from fix" -l dry-run -d 'dry run; don\'t build anything'
complete -c x.py -n "__fish_seen_subcommand_from fix" -l json-output -d 'use message-format=json'
complete -c x.py -n "__fish_seen_subcommand_from fix" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x.py -n "__fish_seen_subcommand_from fix" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x.py -n "__fish_seen_subcommand_from fix" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l config -d 'TOML configuration file for build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l build-dir -d 'Build directory, overrides `build.build-dir` in `config.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l build -d 'build target of the stage0 compiler' -r -f
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l host -d 'host targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l target -d 'target targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l exclude -d 'build paths to exclude' -r -F
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l skip -d 'build paths to skip' -r -F
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l rustc-error-format -r -f
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from fmt" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny	'',warn	'',default	''}"
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l error-format -d 'rustc error format' -r -f
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always	'',never	'',auto	''}"
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l llvm-skip-rebuild -d 'whether rebuilding llvm should be skipped, overriding `skip-rebuld` in config.toml' -r -f -a "{true	'',false	''}"
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l set -d 'override options in config.toml' -r -f
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l check -d 'check formatting instead of applying'
complete -c x.py -n "__fish_seen_subcommand_from fmt" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x.py -n "__fish_seen_subcommand_from fmt" -s i -l incremental -d 'use incremental compilation'
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l dry-run -d 'dry run; don\'t build anything'
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l json-output -d 'use message-format=json'
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x.py -n "__fish_seen_subcommand_from fmt" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x.py -n "__fish_seen_subcommand_from fmt" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x.py -n "__fish_seen_subcommand_from doc" -l config -d 'TOML configuration file for build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from doc" -l build-dir -d 'Build directory, overrides `build.build-dir` in `config.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from doc" -l build -d 'build target of the stage0 compiler' -r -f
complete -c x.py -n "__fish_seen_subcommand_from doc" -l host -d 'host targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from doc" -l target -d 'target targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from doc" -l exclude -d 'build paths to exclude' -r -F
complete -c x.py -n "__fish_seen_subcommand_from doc" -l skip -d 'build paths to skip' -r -F
complete -c x.py -n "__fish_seen_subcommand_from doc" -l rustc-error-format -r -f
complete -c x.py -n "__fish_seen_subcommand_from doc" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x.py -n "__fish_seen_subcommand_from doc" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from doc" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from doc" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from doc" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from doc" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x.py -n "__fish_seen_subcommand_from doc" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny	'',warn	'',default	''}"
complete -c x.py -n "__fish_seen_subcommand_from doc" -l error-format -d 'rustc error format' -r -f
complete -c x.py -n "__fish_seen_subcommand_from doc" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always	'',never	'',auto	''}"
complete -c x.py -n "__fish_seen_subcommand_from doc" -l llvm-skip-rebuild -d 'whether rebuilding llvm should be skipped, overriding `skip-rebuld` in config.toml' -r -f -a "{true	'',false	''}"
complete -c x.py -n "__fish_seen_subcommand_from doc" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from doc" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from doc" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from doc" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x.py -n "__fish_seen_subcommand_from doc" -l set -d 'override options in config.toml' -r -f
complete -c x.py -n "__fish_seen_subcommand_from doc" -l open -d 'open the docs in a browser'
complete -c x.py -n "__fish_seen_subcommand_from doc" -l json -d 'render the documentation in JSON format in addition to the usual HTML format'
complete -c x.py -n "__fish_seen_subcommand_from doc" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x.py -n "__fish_seen_subcommand_from doc" -s i -l incremental -d 'use incremental compilation'
complete -c x.py -n "__fish_seen_subcommand_from doc" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x.py -n "__fish_seen_subcommand_from doc" -l dry-run -d 'dry run; don\'t build anything'
complete -c x.py -n "__fish_seen_subcommand_from doc" -l json-output -d 'use message-format=json'
complete -c x.py -n "__fish_seen_subcommand_from doc" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x.py -n "__fish_seen_subcommand_from doc" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x.py -n "__fish_seen_subcommand_from doc" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x.py -n "__fish_seen_subcommand_from test" -l skip -d 'skips tests matching SUBSTRING, if supported by test tool. May be passed multiple times' -r -F
complete -c x.py -n "__fish_seen_subcommand_from test" -l test-args -d 'extra arguments to be passed for the test tool being used (e.g. libtest, compiletest or rustdoc)' -r
complete -c x.py -n "__fish_seen_subcommand_from test" -l rustc-args -d 'extra options to pass the compiler when running tests' -r
complete -c x.py -n "__fish_seen_subcommand_from test" -l extra-checks -d 'comma-separated list of other files types to check (accepts py, py:lint, py:fmt, shell)' -r
complete -c x.py -n "__fish_seen_subcommand_from test" -l compare-mode -d 'mode describing what file the actual ui output will be compared to' -r
complete -c x.py -n "__fish_seen_subcommand_from test" -l pass -d 'force {check,build,run}-pass tests to this mode' -r
complete -c x.py -n "__fish_seen_subcommand_from test" -l run -d 'whether to execute run-* tests' -r
complete -c x.py -n "__fish_seen_subcommand_from test" -l config -d 'TOML configuration file for build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from test" -l build-dir -d 'Build directory, overrides `build.build-dir` in `config.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from test" -l build -d 'build target of the stage0 compiler' -r -f
complete -c x.py -n "__fish_seen_subcommand_from test" -l host -d 'host targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from test" -l target -d 'target targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from test" -l exclude -d 'build paths to exclude' -r -F
complete -c x.py -n "__fish_seen_subcommand_from test" -l rustc-error-format -r -f
complete -c x.py -n "__fish_seen_subcommand_from test" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x.py -n "__fish_seen_subcommand_from test" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from test" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from test" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from test" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from test" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x.py -n "__fish_seen_subcommand_from test" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny	'',warn	'',default	''}"
complete -c x.py -n "__fish_seen_subcommand_from test" -l error-format -d 'rustc error format' -r -f
complete -c x.py -n "__fish_seen_subcommand_from test" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always	'',never	'',auto	''}"
complete -c x.py -n "__fish_seen_subcommand_from test" -l llvm-skip-rebuild -d 'whether rebuilding llvm should be skipped, overriding `skip-rebuld` in config.toml' -r -f -a "{true	'',false	''}"
complete -c x.py -n "__fish_seen_subcommand_from test" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from test" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from test" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from test" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x.py -n "__fish_seen_subcommand_from test" -l set -d 'override options in config.toml' -r -f
complete -c x.py -n "__fish_seen_subcommand_from test" -l no-fail-fast -d 'run all tests regardless of failure'
complete -c x.py -n "__fish_seen_subcommand_from test" -l no-doc -d 'do not run doc tests'
complete -c x.py -n "__fish_seen_subcommand_from test" -l doc -d 'only run doc tests'
complete -c x.py -n "__fish_seen_subcommand_from test" -l bless -d 'whether to automatically update stderr/stdout files'
complete -c x.py -n "__fish_seen_subcommand_from test" -l force-rerun -d 'rerun tests even if the inputs are unchanged'
complete -c x.py -n "__fish_seen_subcommand_from test" -l only-modified -d 'only run tests that result has been changed'
complete -c x.py -n "__fish_seen_subcommand_from test" -l rustfix-coverage -d 'enable this to generate a Rustfix coverage file, which is saved in `/<build_base>/rustfix_missing_coverage.txt`'
complete -c x.py -n "__fish_seen_subcommand_from test" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x.py -n "__fish_seen_subcommand_from test" -s i -l incremental -d 'use incremental compilation'
complete -c x.py -n "__fish_seen_subcommand_from test" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x.py -n "__fish_seen_subcommand_from test" -l dry-run -d 'dry run; don\'t build anything'
complete -c x.py -n "__fish_seen_subcommand_from test" -l json-output -d 'use message-format=json'
complete -c x.py -n "__fish_seen_subcommand_from test" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x.py -n "__fish_seen_subcommand_from test" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x.py -n "__fish_seen_subcommand_from test" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x.py -n "__fish_seen_subcommand_from bench" -l test-args -r
complete -c x.py -n "__fish_seen_subcommand_from bench" -l config -d 'TOML configuration file for build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from bench" -l build-dir -d 'Build directory, overrides `build.build-dir` in `config.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from bench" -l build -d 'build target of the stage0 compiler' -r -f
complete -c x.py -n "__fish_seen_subcommand_from bench" -l host -d 'host targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from bench" -l target -d 'target targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from bench" -l exclude -d 'build paths to exclude' -r -F
complete -c x.py -n "__fish_seen_subcommand_from bench" -l skip -d 'build paths to skip' -r -F
complete -c x.py -n "__fish_seen_subcommand_from bench" -l rustc-error-format -r -f
complete -c x.py -n "__fish_seen_subcommand_from bench" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x.py -n "__fish_seen_subcommand_from bench" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from bench" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from bench" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from bench" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from bench" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x.py -n "__fish_seen_subcommand_from bench" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny	'',warn	'',default	''}"
complete -c x.py -n "__fish_seen_subcommand_from bench" -l error-format -d 'rustc error format' -r -f
complete -c x.py -n "__fish_seen_subcommand_from bench" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always	'',never	'',auto	''}"
complete -c x.py -n "__fish_seen_subcommand_from bench" -l llvm-skip-rebuild -d 'whether rebuilding llvm should be skipped, overriding `skip-rebuld` in config.toml' -r -f -a "{true	'',false	''}"
complete -c x.py -n "__fish_seen_subcommand_from bench" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from bench" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from bench" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from bench" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x.py -n "__fish_seen_subcommand_from bench" -l set -d 'override options in config.toml' -r -f
complete -c x.py -n "__fish_seen_subcommand_from bench" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x.py -n "__fish_seen_subcommand_from bench" -s i -l incremental -d 'use incremental compilation'
complete -c x.py -n "__fish_seen_subcommand_from bench" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x.py -n "__fish_seen_subcommand_from bench" -l dry-run -d 'dry run; don\'t build anything'
complete -c x.py -n "__fish_seen_subcommand_from bench" -l json-output -d 'use message-format=json'
complete -c x.py -n "__fish_seen_subcommand_from bench" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x.py -n "__fish_seen_subcommand_from bench" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x.py -n "__fish_seen_subcommand_from bench" -s h -l help -d 'Print help'
complete -c x.py -n "__fish_seen_subcommand_from clean" -l stage -d 'Clean a specific stage without touching other artifacts. By default, every stage is cleaned if this option is not used' -r
complete -c x.py -n "__fish_seen_subcommand_from clean" -l config -d 'TOML configuration file for build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from clean" -l build-dir -d 'Build directory, overrides `build.build-dir` in `config.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from clean" -l build -d 'build target of the stage0 compiler' -r -f
complete -c x.py -n "__fish_seen_subcommand_from clean" -l host -d 'host targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from clean" -l target -d 'target targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from clean" -l exclude -d 'build paths to exclude' -r -F
complete -c x.py -n "__fish_seen_subcommand_from clean" -l skip -d 'build paths to skip' -r -F
complete -c x.py -n "__fish_seen_subcommand_from clean" -l rustc-error-format -r -f
complete -c x.py -n "__fish_seen_subcommand_from clean" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x.py -n "__fish_seen_subcommand_from clean" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from clean" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from clean" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from clean" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x.py -n "__fish_seen_subcommand_from clean" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny	'',warn	'',default	''}"
complete -c x.py -n "__fish_seen_subcommand_from clean" -l error-format -d 'rustc error format' -r -f
complete -c x.py -n "__fish_seen_subcommand_from clean" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always	'',never	'',auto	''}"
complete -c x.py -n "__fish_seen_subcommand_from clean" -l llvm-skip-rebuild -d 'whether rebuilding llvm should be skipped, overriding `skip-rebuld` in config.toml' -r -f -a "{true	'',false	''}"
complete -c x.py -n "__fish_seen_subcommand_from clean" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from clean" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from clean" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from clean" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x.py -n "__fish_seen_subcommand_from clean" -l set -d 'override options in config.toml' -r -f
complete -c x.py -n "__fish_seen_subcommand_from clean" -l all -d 'Clean the entire build directory (not used by default)'
complete -c x.py -n "__fish_seen_subcommand_from clean" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x.py -n "__fish_seen_subcommand_from clean" -s i -l incremental -d 'use incremental compilation'
complete -c x.py -n "__fish_seen_subcommand_from clean" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x.py -n "__fish_seen_subcommand_from clean" -l dry-run -d 'dry run; don\'t build anything'
complete -c x.py -n "__fish_seen_subcommand_from clean" -l json-output -d 'use message-format=json'
complete -c x.py -n "__fish_seen_subcommand_from clean" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x.py -n "__fish_seen_subcommand_from clean" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x.py -n "__fish_seen_subcommand_from clean" -s h -l help -d 'Print help'
complete -c x.py -n "__fish_seen_subcommand_from dist" -l config -d 'TOML configuration file for build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from dist" -l build-dir -d 'Build directory, overrides `build.build-dir` in `config.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from dist" -l build -d 'build target of the stage0 compiler' -r -f
complete -c x.py -n "__fish_seen_subcommand_from dist" -l host -d 'host targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from dist" -l target -d 'target targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from dist" -l exclude -d 'build paths to exclude' -r -F
complete -c x.py -n "__fish_seen_subcommand_from dist" -l skip -d 'build paths to skip' -r -F
complete -c x.py -n "__fish_seen_subcommand_from dist" -l rustc-error-format -r -f
complete -c x.py -n "__fish_seen_subcommand_from dist" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x.py -n "__fish_seen_subcommand_from dist" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from dist" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from dist" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from dist" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from dist" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x.py -n "__fish_seen_subcommand_from dist" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny	'',warn	'',default	''}"
complete -c x.py -n "__fish_seen_subcommand_from dist" -l error-format -d 'rustc error format' -r -f
complete -c x.py -n "__fish_seen_subcommand_from dist" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always	'',never	'',auto	''}"
complete -c x.py -n "__fish_seen_subcommand_from dist" -l llvm-skip-rebuild -d 'whether rebuilding llvm should be skipped, overriding `skip-rebuld` in config.toml' -r -f -a "{true	'',false	''}"
complete -c x.py -n "__fish_seen_subcommand_from dist" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from dist" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from dist" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from dist" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x.py -n "__fish_seen_subcommand_from dist" -l set -d 'override options in config.toml' -r -f
complete -c x.py -n "__fish_seen_subcommand_from dist" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x.py -n "__fish_seen_subcommand_from dist" -s i -l incremental -d 'use incremental compilation'
complete -c x.py -n "__fish_seen_subcommand_from dist" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x.py -n "__fish_seen_subcommand_from dist" -l dry-run -d 'dry run; don\'t build anything'
complete -c x.py -n "__fish_seen_subcommand_from dist" -l json-output -d 'use message-format=json'
complete -c x.py -n "__fish_seen_subcommand_from dist" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x.py -n "__fish_seen_subcommand_from dist" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x.py -n "__fish_seen_subcommand_from dist" -s h -l help -d 'Print help'
complete -c x.py -n "__fish_seen_subcommand_from install" -l config -d 'TOML configuration file for build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from install" -l build-dir -d 'Build directory, overrides `build.build-dir` in `config.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from install" -l build -d 'build target of the stage0 compiler' -r -f
complete -c x.py -n "__fish_seen_subcommand_from install" -l host -d 'host targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from install" -l target -d 'target targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from install" -l exclude -d 'build paths to exclude' -r -F
complete -c x.py -n "__fish_seen_subcommand_from install" -l skip -d 'build paths to skip' -r -F
complete -c x.py -n "__fish_seen_subcommand_from install" -l rustc-error-format -r -f
complete -c x.py -n "__fish_seen_subcommand_from install" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x.py -n "__fish_seen_subcommand_from install" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from install" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from install" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from install" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from install" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x.py -n "__fish_seen_subcommand_from install" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny	'',warn	'',default	''}"
complete -c x.py -n "__fish_seen_subcommand_from install" -l error-format -d 'rustc error format' -r -f
complete -c x.py -n "__fish_seen_subcommand_from install" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always	'',never	'',auto	''}"
complete -c x.py -n "__fish_seen_subcommand_from install" -l llvm-skip-rebuild -d 'whether rebuilding llvm should be skipped, overriding `skip-rebuld` in config.toml' -r -f -a "{true	'',false	''}"
complete -c x.py -n "__fish_seen_subcommand_from install" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from install" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from install" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from install" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x.py -n "__fish_seen_subcommand_from install" -l set -d 'override options in config.toml' -r -f
complete -c x.py -n "__fish_seen_subcommand_from install" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x.py -n "__fish_seen_subcommand_from install" -s i -l incremental -d 'use incremental compilation'
complete -c x.py -n "__fish_seen_subcommand_from install" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x.py -n "__fish_seen_subcommand_from install" -l dry-run -d 'dry run; don\'t build anything'
complete -c x.py -n "__fish_seen_subcommand_from install" -l json-output -d 'use message-format=json'
complete -c x.py -n "__fish_seen_subcommand_from install" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x.py -n "__fish_seen_subcommand_from install" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x.py -n "__fish_seen_subcommand_from install" -s h -l help -d 'Print help'
complete -c x.py -n "__fish_seen_subcommand_from run" -l args -d 'arguments for the tool' -r
complete -c x.py -n "__fish_seen_subcommand_from run" -l config -d 'TOML configuration file for build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from run" -l build-dir -d 'Build directory, overrides `build.build-dir` in `config.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from run" -l build -d 'build target of the stage0 compiler' -r -f
complete -c x.py -n "__fish_seen_subcommand_from run" -l host -d 'host targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from run" -l target -d 'target targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from run" -l exclude -d 'build paths to exclude' -r -F
complete -c x.py -n "__fish_seen_subcommand_from run" -l skip -d 'build paths to skip' -r -F
complete -c x.py -n "__fish_seen_subcommand_from run" -l rustc-error-format -r -f
complete -c x.py -n "__fish_seen_subcommand_from run" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x.py -n "__fish_seen_subcommand_from run" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from run" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from run" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from run" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from run" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x.py -n "__fish_seen_subcommand_from run" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny	'',warn	'',default	''}"
complete -c x.py -n "__fish_seen_subcommand_from run" -l error-format -d 'rustc error format' -r -f
complete -c x.py -n "__fish_seen_subcommand_from run" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always	'',never	'',auto	''}"
complete -c x.py -n "__fish_seen_subcommand_from run" -l llvm-skip-rebuild -d 'whether rebuilding llvm should be skipped, overriding `skip-rebuld` in config.toml' -r -f -a "{true	'',false	''}"
complete -c x.py -n "__fish_seen_subcommand_from run" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from run" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from run" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from run" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x.py -n "__fish_seen_subcommand_from run" -l set -d 'override options in config.toml' -r -f
complete -c x.py -n "__fish_seen_subcommand_from run" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x.py -n "__fish_seen_subcommand_from run" -s i -l incremental -d 'use incremental compilation'
complete -c x.py -n "__fish_seen_subcommand_from run" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x.py -n "__fish_seen_subcommand_from run" -l dry-run -d 'dry run; don\'t build anything'
complete -c x.py -n "__fish_seen_subcommand_from run" -l json-output -d 'use message-format=json'
complete -c x.py -n "__fish_seen_subcommand_from run" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x.py -n "__fish_seen_subcommand_from run" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x.py -n "__fish_seen_subcommand_from run" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x.py -n "__fish_seen_subcommand_from setup" -l config -d 'TOML configuration file for build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from setup" -l build-dir -d 'Build directory, overrides `build.build-dir` in `config.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from setup" -l build -d 'build target of the stage0 compiler' -r -f
complete -c x.py -n "__fish_seen_subcommand_from setup" -l host -d 'host targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from setup" -l target -d 'target targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from setup" -l exclude -d 'build paths to exclude' -r -F
complete -c x.py -n "__fish_seen_subcommand_from setup" -l skip -d 'build paths to skip' -r -F
complete -c x.py -n "__fish_seen_subcommand_from setup" -l rustc-error-format -r -f
complete -c x.py -n "__fish_seen_subcommand_from setup" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x.py -n "__fish_seen_subcommand_from setup" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from setup" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from setup" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from setup" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from setup" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x.py -n "__fish_seen_subcommand_from setup" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny	'',warn	'',default	''}"
complete -c x.py -n "__fish_seen_subcommand_from setup" -l error-format -d 'rustc error format' -r -f
complete -c x.py -n "__fish_seen_subcommand_from setup" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always	'',never	'',auto	''}"
complete -c x.py -n "__fish_seen_subcommand_from setup" -l llvm-skip-rebuild -d 'whether rebuilding llvm should be skipped, overriding `skip-rebuld` in config.toml' -r -f -a "{true	'',false	''}"
complete -c x.py -n "__fish_seen_subcommand_from setup" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from setup" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from setup" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from setup" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x.py -n "__fish_seen_subcommand_from setup" -l set -d 'override options in config.toml' -r -f
complete -c x.py -n "__fish_seen_subcommand_from setup" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x.py -n "__fish_seen_subcommand_from setup" -s i -l incremental -d 'use incremental compilation'
complete -c x.py -n "__fish_seen_subcommand_from setup" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x.py -n "__fish_seen_subcommand_from setup" -l dry-run -d 'dry run; don\'t build anything'
complete -c x.py -n "__fish_seen_subcommand_from setup" -l json-output -d 'use message-format=json'
complete -c x.py -n "__fish_seen_subcommand_from setup" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x.py -n "__fish_seen_subcommand_from setup" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x.py -n "__fish_seen_subcommand_from setup" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l config -d 'TOML configuration file for build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l build-dir -d 'Build directory, overrides `build.build-dir` in `config.toml`' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l build -d 'build target of the stage0 compiler' -r -f
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l host -d 'host targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l target -d 'target targets to build' -r -f
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l exclude -d 'build paths to exclude' -r -F
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l skip -d 'build paths to skip' -r -F
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l rustc-error-format -r -f
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l on-fail -d 'command to run on failure' -r -f -a "(__fish_complete_command)"
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l stage -d 'stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l keep-stage -d 'stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l keep-stage-std -d 'stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)' -r -f
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l src -d 'path to the root of the rust checkout' -r -f -a "(__fish_complete_directories)"
complete -c x.py -n "__fish_seen_subcommand_from suggest" -s j -l jobs -d 'number of jobs to run in parallel' -r -f
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l warnings -d 'if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour' -r -f -a "{deny	'',warn	'',default	''}"
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l error-format -d 'rustc error format' -r -f
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l color -d 'whether to use color in cargo and rustc output' -r -f -a "{always	'',never	'',auto	''}"
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l llvm-skip-rebuild -d 'whether rebuilding llvm should be skipped, overriding `skip-rebuld` in config.toml' -r -f -a "{true	'',false	''}"
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l rust-profile-generate -d 'generate PGO profile with rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l rust-profile-use -d 'use PGO profile for rustc build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l llvm-profile-use -d 'use PGO profile for LLVM build' -r -F
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l reproducible-artifact -d 'Additional reproducible artifacts that should be added to the reproducible artifacts archive' -r
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l set -d 'override options in config.toml' -r -f
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l run -d 'run suggested tests'
complete -c x.py -n "__fish_seen_subcommand_from suggest" -s v -l verbose -d 'use verbose output (-vv for very verbose)'
complete -c x.py -n "__fish_seen_subcommand_from suggest" -s i -l incremental -d 'use incremental compilation'
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l include-default-paths -d 'include default paths in addition to the provided ones'
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l dry-run -d 'dry run; don\'t build anything'
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l json-output -d 'use message-format=json'
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l llvm-profile-generate -d 'generate PGO profile with llvm built for rustc'
complete -c x.py -n "__fish_seen_subcommand_from suggest" -l enable-bolt-settings -d 'Enable BOLT link flags'
complete -c x.py -n "__fish_seen_subcommand_from suggest" -s h -l help -d 'Print help (see more with \'--help\')'
