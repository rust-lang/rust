#compdef x.py

autoload -U is-at-least

_x.py() {
    typeset -A opt_args
    typeset -a _arguments_options
    local ret=1

    if is-at-least 5.2; then
        _arguments_options=(-s -S -C)
    else
        _arguments_options=(-s -C)
    fi

    local context curcontext="$curcontext" state line
    _arguments "${_arguments_options[@]}" : \
'--config=[TOML configuration file for build]:FILE:_files' \
'--build-dir=[Build directory, overrides \`build.build-dir\` in \`config.toml\`]:DIR:_files -/' \
'--build=[build target of the stage0 compiler]:BUILD:( )' \
'--host=[host targets to build]:HOST:( )' \
'--target=[target targets to build]:TARGET:( )' \
'*--exclude=[build paths to exclude]:PATH:_files' \
'*--skip=[build paths to skip]:PATH:_files' \
'--rustc-error-format=[]:RUSTC_ERROR_FORMAT:( )' \
'--on-fail=[command to run on failure]:CMD:_cmdstring' \
'--stage=[stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)]:N:( )' \
'*--keep-stage=[stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'*--keep-stage-std=[stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'--src=[path to the root of the rust checkout]:DIR:_files -/' \
'-j+[number of jobs to run in parallel]:JOBS:( )' \
'--jobs=[number of jobs to run in parallel]:JOBS:( )' \
'--warnings=[if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour]:deny|warn:(deny warn default)' \
'--error-format=[rustc error format]:FORMAT:( )' \
'--color=[whether to use color in cargo and rustc output]:STYLE:(always never auto)' \
'--rust-profile-generate=[generate PGO profile with rustc build]:PROFILE:_files' \
'--rust-profile-use=[use PGO profile for rustc build]:PROFILE:_files' \
'--llvm-profile-use=[use PGO profile for LLVM build]:PROFILE:_files' \
'*--reproducible-artifact=[Additional reproducible artifacts that should be added to the reproducible artifacts archive]:REPRODUCIBLE_ARTIFACT: ' \
'*--set=[override options in config.toml]:section.option=value:( )' \
'*-v[use verbose output (-vv for very verbose)]' \
'*--verbose[use verbose output (-vv for very verbose)]' \
'-i[use incremental compilation]' \
'--incremental[use incremental compilation]' \
'--include-default-paths[include default paths in addition to the provided ones]' \
'--dry-run[dry run; don'\''t build anything]' \
'--dump-bootstrap-shims[Indicates whether to dump the work done from bootstrap shims]' \
'--json-output[use message-format=json]' \
'--bypass-bootstrap-lock[Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)]' \
'--llvm-profile-generate[generate PGO profile with llvm built for rustc]' \
'--enable-bolt-settings[Enable BOLT link flags]' \
'--skip-stage0-validation[Skip stage0 compiler validation]' \
'-h[Print help (see more with '\''--help'\'')]' \
'--help[Print help (see more with '\''--help'\'')]' \
'::paths -- paths for the subcommand:_files' \
'::free_args -- arguments passed to subcommands:' \
":: :_x.py_commands" \
"*::: :->bootstrap" \
&& ret=0
    case $state in
    (bootstrap)
        words=($line[3] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:x.py-command-$line[3]:"
        case $line[3] in
            (build)
_arguments "${_arguments_options[@]}" : \
'--config=[TOML configuration file for build]:FILE:_files' \
'--build-dir=[Build directory, overrides \`build.build-dir\` in \`config.toml\`]:DIR:_files -/' \
'--build=[build target of the stage0 compiler]:BUILD:( )' \
'--host=[host targets to build]:HOST:( )' \
'--target=[target targets to build]:TARGET:( )' \
'*--exclude=[build paths to exclude]:PATH:_files' \
'*--skip=[build paths to skip]:PATH:_files' \
'--rustc-error-format=[]:RUSTC_ERROR_FORMAT:( )' \
'--on-fail=[command to run on failure]:CMD:_cmdstring' \
'--stage=[stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)]:N:( )' \
'*--keep-stage=[stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'*--keep-stage-std=[stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'--src=[path to the root of the rust checkout]:DIR:_files -/' \
'-j+[number of jobs to run in parallel]:JOBS:( )' \
'--jobs=[number of jobs to run in parallel]:JOBS:( )' \
'--warnings=[if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour]:deny|warn:(deny warn default)' \
'--error-format=[rustc error format]:FORMAT:( )' \
'--color=[whether to use color in cargo and rustc output]:STYLE:(always never auto)' \
'--rust-profile-generate=[generate PGO profile with rustc build]:PROFILE:_files' \
'--rust-profile-use=[use PGO profile for rustc build]:PROFILE:_files' \
'--llvm-profile-use=[use PGO profile for LLVM build]:PROFILE:_files' \
'*--reproducible-artifact=[Additional reproducible artifacts that should be added to the reproducible artifacts archive]:REPRODUCIBLE_ARTIFACT: ' \
'*--set=[override options in config.toml]:section.option=value:( )' \
'*-v[use verbose output (-vv for very verbose)]' \
'*--verbose[use verbose output (-vv for very verbose)]' \
'-i[use incremental compilation]' \
'--incremental[use incremental compilation]' \
'--include-default-paths[include default paths in addition to the provided ones]' \
'--dry-run[dry run; don'\''t build anything]' \
'--dump-bootstrap-shims[Indicates whether to dump the work done from bootstrap shims]' \
'--json-output[use message-format=json]' \
'--bypass-bootstrap-lock[Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)]' \
'--llvm-profile-generate[generate PGO profile with llvm built for rustc]' \
'--enable-bolt-settings[Enable BOLT link flags]' \
'--skip-stage0-validation[Skip stage0 compiler validation]' \
'-h[Print help (see more with '\''--help'\'')]' \
'--help[Print help (see more with '\''--help'\'')]' \
'*::paths -- paths for the subcommand:_files' \
&& ret=0
;;
(check)
_arguments "${_arguments_options[@]}" : \
'--config=[TOML configuration file for build]:FILE:_files' \
'--build-dir=[Build directory, overrides \`build.build-dir\` in \`config.toml\`]:DIR:_files -/' \
'--build=[build target of the stage0 compiler]:BUILD:( )' \
'--host=[host targets to build]:HOST:( )' \
'--target=[target targets to build]:TARGET:( )' \
'*--exclude=[build paths to exclude]:PATH:_files' \
'*--skip=[build paths to skip]:PATH:_files' \
'--rustc-error-format=[]:RUSTC_ERROR_FORMAT:( )' \
'--on-fail=[command to run on failure]:CMD:_cmdstring' \
'--stage=[stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)]:N:( )' \
'*--keep-stage=[stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'*--keep-stage-std=[stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'--src=[path to the root of the rust checkout]:DIR:_files -/' \
'-j+[number of jobs to run in parallel]:JOBS:( )' \
'--jobs=[number of jobs to run in parallel]:JOBS:( )' \
'--warnings=[if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour]:deny|warn:(deny warn default)' \
'--error-format=[rustc error format]:FORMAT:( )' \
'--color=[whether to use color in cargo and rustc output]:STYLE:(always never auto)' \
'--rust-profile-generate=[generate PGO profile with rustc build]:PROFILE:_files' \
'--rust-profile-use=[use PGO profile for rustc build]:PROFILE:_files' \
'--llvm-profile-use=[use PGO profile for LLVM build]:PROFILE:_files' \
'*--reproducible-artifact=[Additional reproducible artifacts that should be added to the reproducible artifacts archive]:REPRODUCIBLE_ARTIFACT: ' \
'*--set=[override options in config.toml]:section.option=value:( )' \
'--all-targets[Check all targets]' \
'*-v[use verbose output (-vv for very verbose)]' \
'*--verbose[use verbose output (-vv for very verbose)]' \
'-i[use incremental compilation]' \
'--incremental[use incremental compilation]' \
'--include-default-paths[include default paths in addition to the provided ones]' \
'--dry-run[dry run; don'\''t build anything]' \
'--dump-bootstrap-shims[Indicates whether to dump the work done from bootstrap shims]' \
'--json-output[use message-format=json]' \
'--bypass-bootstrap-lock[Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)]' \
'--llvm-profile-generate[generate PGO profile with llvm built for rustc]' \
'--enable-bolt-settings[Enable BOLT link flags]' \
'--skip-stage0-validation[Skip stage0 compiler validation]' \
'-h[Print help (see more with '\''--help'\'')]' \
'--help[Print help (see more with '\''--help'\'')]' \
'*::paths -- paths for the subcommand:_files' \
&& ret=0
;;
(clippy)
_arguments "${_arguments_options[@]}" : \
'*-A+[clippy lints to allow]:LINT: ' \
'*-D+[clippy lints to deny]:LINT: ' \
'*-W+[clippy lints to warn on]:LINT: ' \
'*-F+[clippy lints to forbid]:LINT: ' \
'--config=[TOML configuration file for build]:FILE:_files' \
'--build-dir=[Build directory, overrides \`build.build-dir\` in \`config.toml\`]:DIR:_files -/' \
'--build=[build target of the stage0 compiler]:BUILD:( )' \
'--host=[host targets to build]:HOST:( )' \
'--target=[target targets to build]:TARGET:( )' \
'*--exclude=[build paths to exclude]:PATH:_files' \
'*--skip=[build paths to skip]:PATH:_files' \
'--rustc-error-format=[]:RUSTC_ERROR_FORMAT:( )' \
'--on-fail=[command to run on failure]:CMD:_cmdstring' \
'--stage=[stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)]:N:( )' \
'*--keep-stage=[stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'*--keep-stage-std=[stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'--src=[path to the root of the rust checkout]:DIR:_files -/' \
'-j+[number of jobs to run in parallel]:JOBS:( )' \
'--jobs=[number of jobs to run in parallel]:JOBS:( )' \
'--warnings=[if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour]:deny|warn:(deny warn default)' \
'--error-format=[rustc error format]:FORMAT:( )' \
'--color=[whether to use color in cargo and rustc output]:STYLE:(always never auto)' \
'--rust-profile-generate=[generate PGO profile with rustc build]:PROFILE:_files' \
'--rust-profile-use=[use PGO profile for rustc build]:PROFILE:_files' \
'--llvm-profile-use=[use PGO profile for LLVM build]:PROFILE:_files' \
'*--reproducible-artifact=[Additional reproducible artifacts that should be added to the reproducible artifacts archive]:REPRODUCIBLE_ARTIFACT: ' \
'*--set=[override options in config.toml]:section.option=value:( )' \
'--fix[]' \
'--allow-dirty[]' \
'--allow-staged[]' \
'*-v[use verbose output (-vv for very verbose)]' \
'*--verbose[use verbose output (-vv for very verbose)]' \
'-i[use incremental compilation]' \
'--incremental[use incremental compilation]' \
'--include-default-paths[include default paths in addition to the provided ones]' \
'--dry-run[dry run; don'\''t build anything]' \
'--dump-bootstrap-shims[Indicates whether to dump the work done from bootstrap shims]' \
'--json-output[use message-format=json]' \
'--bypass-bootstrap-lock[Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)]' \
'--llvm-profile-generate[generate PGO profile with llvm built for rustc]' \
'--enable-bolt-settings[Enable BOLT link flags]' \
'--skip-stage0-validation[Skip stage0 compiler validation]' \
'-h[Print help (see more with '\''--help'\'')]' \
'--help[Print help (see more with '\''--help'\'')]' \
'*::paths -- paths for the subcommand:_files' \
&& ret=0
;;
(fix)
_arguments "${_arguments_options[@]}" : \
'--config=[TOML configuration file for build]:FILE:_files' \
'--build-dir=[Build directory, overrides \`build.build-dir\` in \`config.toml\`]:DIR:_files -/' \
'--build=[build target of the stage0 compiler]:BUILD:( )' \
'--host=[host targets to build]:HOST:( )' \
'--target=[target targets to build]:TARGET:( )' \
'*--exclude=[build paths to exclude]:PATH:_files' \
'*--skip=[build paths to skip]:PATH:_files' \
'--rustc-error-format=[]:RUSTC_ERROR_FORMAT:( )' \
'--on-fail=[command to run on failure]:CMD:_cmdstring' \
'--stage=[stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)]:N:( )' \
'*--keep-stage=[stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'*--keep-stage-std=[stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'--src=[path to the root of the rust checkout]:DIR:_files -/' \
'-j+[number of jobs to run in parallel]:JOBS:( )' \
'--jobs=[number of jobs to run in parallel]:JOBS:( )' \
'--warnings=[if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour]:deny|warn:(deny warn default)' \
'--error-format=[rustc error format]:FORMAT:( )' \
'--color=[whether to use color in cargo and rustc output]:STYLE:(always never auto)' \
'--rust-profile-generate=[generate PGO profile with rustc build]:PROFILE:_files' \
'--rust-profile-use=[use PGO profile for rustc build]:PROFILE:_files' \
'--llvm-profile-use=[use PGO profile for LLVM build]:PROFILE:_files' \
'*--reproducible-artifact=[Additional reproducible artifacts that should be added to the reproducible artifacts archive]:REPRODUCIBLE_ARTIFACT: ' \
'*--set=[override options in config.toml]:section.option=value:( )' \
'*-v[use verbose output (-vv for very verbose)]' \
'*--verbose[use verbose output (-vv for very verbose)]' \
'-i[use incremental compilation]' \
'--incremental[use incremental compilation]' \
'--include-default-paths[include default paths in addition to the provided ones]' \
'--dry-run[dry run; don'\''t build anything]' \
'--dump-bootstrap-shims[Indicates whether to dump the work done from bootstrap shims]' \
'--json-output[use message-format=json]' \
'--bypass-bootstrap-lock[Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)]' \
'--llvm-profile-generate[generate PGO profile with llvm built for rustc]' \
'--enable-bolt-settings[Enable BOLT link flags]' \
'--skip-stage0-validation[Skip stage0 compiler validation]' \
'-h[Print help (see more with '\''--help'\'')]' \
'--help[Print help (see more with '\''--help'\'')]' \
'*::paths -- paths for the subcommand:_files' \
&& ret=0
;;
(fmt)
_arguments "${_arguments_options[@]}" : \
'--config=[TOML configuration file for build]:FILE:_files' \
'--build-dir=[Build directory, overrides \`build.build-dir\` in \`config.toml\`]:DIR:_files -/' \
'--build=[build target of the stage0 compiler]:BUILD:( )' \
'--host=[host targets to build]:HOST:( )' \
'--target=[target targets to build]:TARGET:( )' \
'*--exclude=[build paths to exclude]:PATH:_files' \
'*--skip=[build paths to skip]:PATH:_files' \
'--rustc-error-format=[]:RUSTC_ERROR_FORMAT:( )' \
'--on-fail=[command to run on failure]:CMD:_cmdstring' \
'--stage=[stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)]:N:( )' \
'*--keep-stage=[stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'*--keep-stage-std=[stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'--src=[path to the root of the rust checkout]:DIR:_files -/' \
'-j+[number of jobs to run in parallel]:JOBS:( )' \
'--jobs=[number of jobs to run in parallel]:JOBS:( )' \
'--warnings=[if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour]:deny|warn:(deny warn default)' \
'--error-format=[rustc error format]:FORMAT:( )' \
'--color=[whether to use color in cargo and rustc output]:STYLE:(always never auto)' \
'--rust-profile-generate=[generate PGO profile with rustc build]:PROFILE:_files' \
'--rust-profile-use=[use PGO profile for rustc build]:PROFILE:_files' \
'--llvm-profile-use=[use PGO profile for LLVM build]:PROFILE:_files' \
'*--reproducible-artifact=[Additional reproducible artifacts that should be added to the reproducible artifacts archive]:REPRODUCIBLE_ARTIFACT: ' \
'*--set=[override options in config.toml]:section.option=value:( )' \
'--check[check formatting instead of applying]' \
'--all[apply to all appropriate files, not just those that have been modified]' \
'*-v[use verbose output (-vv for very verbose)]' \
'*--verbose[use verbose output (-vv for very verbose)]' \
'-i[use incremental compilation]' \
'--incremental[use incremental compilation]' \
'--include-default-paths[include default paths in addition to the provided ones]' \
'--dry-run[dry run; don'\''t build anything]' \
'--dump-bootstrap-shims[Indicates whether to dump the work done from bootstrap shims]' \
'--json-output[use message-format=json]' \
'--bypass-bootstrap-lock[Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)]' \
'--llvm-profile-generate[generate PGO profile with llvm built for rustc]' \
'--enable-bolt-settings[Enable BOLT link flags]' \
'--skip-stage0-validation[Skip stage0 compiler validation]' \
'-h[Print help (see more with '\''--help'\'')]' \
'--help[Print help (see more with '\''--help'\'')]' \
'*::paths -- paths for the subcommand:_files' \
&& ret=0
;;
(doc)
_arguments "${_arguments_options[@]}" : \
'--config=[TOML configuration file for build]:FILE:_files' \
'--build-dir=[Build directory, overrides \`build.build-dir\` in \`config.toml\`]:DIR:_files -/' \
'--build=[build target of the stage0 compiler]:BUILD:( )' \
'--host=[host targets to build]:HOST:( )' \
'--target=[target targets to build]:TARGET:( )' \
'*--exclude=[build paths to exclude]:PATH:_files' \
'*--skip=[build paths to skip]:PATH:_files' \
'--rustc-error-format=[]:RUSTC_ERROR_FORMAT:( )' \
'--on-fail=[command to run on failure]:CMD:_cmdstring' \
'--stage=[stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)]:N:( )' \
'*--keep-stage=[stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'*--keep-stage-std=[stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'--src=[path to the root of the rust checkout]:DIR:_files -/' \
'-j+[number of jobs to run in parallel]:JOBS:( )' \
'--jobs=[number of jobs to run in parallel]:JOBS:( )' \
'--warnings=[if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour]:deny|warn:(deny warn default)' \
'--error-format=[rustc error format]:FORMAT:( )' \
'--color=[whether to use color in cargo and rustc output]:STYLE:(always never auto)' \
'--rust-profile-generate=[generate PGO profile with rustc build]:PROFILE:_files' \
'--rust-profile-use=[use PGO profile for rustc build]:PROFILE:_files' \
'--llvm-profile-use=[use PGO profile for LLVM build]:PROFILE:_files' \
'*--reproducible-artifact=[Additional reproducible artifacts that should be added to the reproducible artifacts archive]:REPRODUCIBLE_ARTIFACT: ' \
'*--set=[override options in config.toml]:section.option=value:( )' \
'--open[open the docs in a browser]' \
'--json[render the documentation in JSON format in addition to the usual HTML format]' \
'*-v[use verbose output (-vv for very verbose)]' \
'*--verbose[use verbose output (-vv for very verbose)]' \
'-i[use incremental compilation]' \
'--incremental[use incremental compilation]' \
'--include-default-paths[include default paths in addition to the provided ones]' \
'--dry-run[dry run; don'\''t build anything]' \
'--dump-bootstrap-shims[Indicates whether to dump the work done from bootstrap shims]' \
'--json-output[use message-format=json]' \
'--bypass-bootstrap-lock[Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)]' \
'--llvm-profile-generate[generate PGO profile with llvm built for rustc]' \
'--enable-bolt-settings[Enable BOLT link flags]' \
'--skip-stage0-validation[Skip stage0 compiler validation]' \
'-h[Print help (see more with '\''--help'\'')]' \
'--help[Print help (see more with '\''--help'\'')]' \
'*::paths -- paths for the subcommand:_files' \
&& ret=0
;;
(test)
_arguments "${_arguments_options[@]}" : \
'*--test-args=[extra arguments to be passed for the test tool being used (e.g. libtest, compiletest or rustdoc)]:ARGS: ' \
'*--compiletest-rustc-args=[extra options to pass the compiler when running compiletest tests]:ARGS: ' \
'--extra-checks=[comma-separated list of other files types to check (accepts py, py\:lint, py\:fmt, shell)]:EXTRA_CHECKS: ' \
'--compare-mode=[mode describing what file the actual ui output will be compared to]:COMPARE MODE: ' \
'--pass=[force {check,build,run}-pass tests to this mode]:check | build | run: ' \
'--run=[whether to execute run-* tests]:auto | always | never: ' \
'--config=[TOML configuration file for build]:FILE:_files' \
'--build-dir=[Build directory, overrides \`build.build-dir\` in \`config.toml\`]:DIR:_files -/' \
'--build=[build target of the stage0 compiler]:BUILD:( )' \
'--host=[host targets to build]:HOST:( )' \
'--target=[target targets to build]:TARGET:( )' \
'*--exclude=[build paths to exclude]:PATH:_files' \
'*--skip=[build paths to skip]:PATH:_files' \
'--rustc-error-format=[]:RUSTC_ERROR_FORMAT:( )' \
'--on-fail=[command to run on failure]:CMD:_cmdstring' \
'--stage=[stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)]:N:( )' \
'*--keep-stage=[stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'*--keep-stage-std=[stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'--src=[path to the root of the rust checkout]:DIR:_files -/' \
'-j+[number of jobs to run in parallel]:JOBS:( )' \
'--jobs=[number of jobs to run in parallel]:JOBS:( )' \
'--warnings=[if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour]:deny|warn:(deny warn default)' \
'--error-format=[rustc error format]:FORMAT:( )' \
'--color=[whether to use color in cargo and rustc output]:STYLE:(always never auto)' \
'--rust-profile-generate=[generate PGO profile with rustc build]:PROFILE:_files' \
'--rust-profile-use=[use PGO profile for rustc build]:PROFILE:_files' \
'--llvm-profile-use=[use PGO profile for LLVM build]:PROFILE:_files' \
'*--reproducible-artifact=[Additional reproducible artifacts that should be added to the reproducible artifacts archive]:REPRODUCIBLE_ARTIFACT: ' \
'*--set=[override options in config.toml]:section.option=value:( )' \
'--no-fail-fast[run all tests regardless of failure]' \
'--no-doc[do not run doc tests]' \
'--doc[only run doc tests]' \
'--bless[whether to automatically update stderr/stdout files]' \
'--force-rerun[rerun tests even if the inputs are unchanged]' \
'--only-modified[only run tests that result has been changed]' \
'--rustfix-coverage[enable this to generate a Rustfix coverage file, which is saved in \`/<build_base>/rustfix_missing_coverage.txt\`]' \
'*-v[use verbose output (-vv for very verbose)]' \
'*--verbose[use verbose output (-vv for very verbose)]' \
'-i[use incremental compilation]' \
'--incremental[use incremental compilation]' \
'--include-default-paths[include default paths in addition to the provided ones]' \
'--dry-run[dry run; don'\''t build anything]' \
'--dump-bootstrap-shims[Indicates whether to dump the work done from bootstrap shims]' \
'--json-output[use message-format=json]' \
'--bypass-bootstrap-lock[Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)]' \
'--llvm-profile-generate[generate PGO profile with llvm built for rustc]' \
'--enable-bolt-settings[Enable BOLT link flags]' \
'--skip-stage0-validation[Skip stage0 compiler validation]' \
'-h[Print help (see more with '\''--help'\'')]' \
'--help[Print help (see more with '\''--help'\'')]' \
'*::paths -- paths for the subcommand:_files' \
&& ret=0
;;
(miri)
_arguments "${_arguments_options[@]}" : \
'*--test-args=[extra arguments to be passed for the test tool being used (e.g. libtest, compiletest or rustdoc)]:ARGS: ' \
'--config=[TOML configuration file for build]:FILE:_files' \
'--build-dir=[Build directory, overrides \`build.build-dir\` in \`config.toml\`]:DIR:_files -/' \
'--build=[build target of the stage0 compiler]:BUILD:( )' \
'--host=[host targets to build]:HOST:( )' \
'--target=[target targets to build]:TARGET:( )' \
'*--exclude=[build paths to exclude]:PATH:_files' \
'*--skip=[build paths to skip]:PATH:_files' \
'--rustc-error-format=[]:RUSTC_ERROR_FORMAT:( )' \
'--on-fail=[command to run on failure]:CMD:_cmdstring' \
'--stage=[stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)]:N:( )' \
'*--keep-stage=[stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'*--keep-stage-std=[stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'--src=[path to the root of the rust checkout]:DIR:_files -/' \
'-j+[number of jobs to run in parallel]:JOBS:( )' \
'--jobs=[number of jobs to run in parallel]:JOBS:( )' \
'--warnings=[if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour]:deny|warn:(deny warn default)' \
'--error-format=[rustc error format]:FORMAT:( )' \
'--color=[whether to use color in cargo and rustc output]:STYLE:(always never auto)' \
'--rust-profile-generate=[generate PGO profile with rustc build]:PROFILE:_files' \
'--rust-profile-use=[use PGO profile for rustc build]:PROFILE:_files' \
'--llvm-profile-use=[use PGO profile for LLVM build]:PROFILE:_files' \
'*--reproducible-artifact=[Additional reproducible artifacts that should be added to the reproducible artifacts archive]:REPRODUCIBLE_ARTIFACT: ' \
'*--set=[override options in config.toml]:section.option=value:( )' \
'--no-fail-fast[run all tests regardless of failure]' \
'--no-doc[do not run doc tests]' \
'--doc[only run doc tests]' \
'*-v[use verbose output (-vv for very verbose)]' \
'*--verbose[use verbose output (-vv for very verbose)]' \
'-i[use incremental compilation]' \
'--incremental[use incremental compilation]' \
'--include-default-paths[include default paths in addition to the provided ones]' \
'--dry-run[dry run; don'\''t build anything]' \
'--dump-bootstrap-shims[Indicates whether to dump the work done from bootstrap shims]' \
'--json-output[use message-format=json]' \
'--bypass-bootstrap-lock[Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)]' \
'--llvm-profile-generate[generate PGO profile with llvm built for rustc]' \
'--enable-bolt-settings[Enable BOLT link flags]' \
'--skip-stage0-validation[Skip stage0 compiler validation]' \
'-h[Print help (see more with '\''--help'\'')]' \
'--help[Print help (see more with '\''--help'\'')]' \
'*::paths -- paths for the subcommand:_files' \
&& ret=0
;;
(bench)
_arguments "${_arguments_options[@]}" : \
'*--test-args=[]:TEST_ARGS: ' \
'--config=[TOML configuration file for build]:FILE:_files' \
'--build-dir=[Build directory, overrides \`build.build-dir\` in \`config.toml\`]:DIR:_files -/' \
'--build=[build target of the stage0 compiler]:BUILD:( )' \
'--host=[host targets to build]:HOST:( )' \
'--target=[target targets to build]:TARGET:( )' \
'*--exclude=[build paths to exclude]:PATH:_files' \
'*--skip=[build paths to skip]:PATH:_files' \
'--rustc-error-format=[]:RUSTC_ERROR_FORMAT:( )' \
'--on-fail=[command to run on failure]:CMD:_cmdstring' \
'--stage=[stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)]:N:( )' \
'*--keep-stage=[stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'*--keep-stage-std=[stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'--src=[path to the root of the rust checkout]:DIR:_files -/' \
'-j+[number of jobs to run in parallel]:JOBS:( )' \
'--jobs=[number of jobs to run in parallel]:JOBS:( )' \
'--warnings=[if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour]:deny|warn:(deny warn default)' \
'--error-format=[rustc error format]:FORMAT:( )' \
'--color=[whether to use color in cargo and rustc output]:STYLE:(always never auto)' \
'--rust-profile-generate=[generate PGO profile with rustc build]:PROFILE:_files' \
'--rust-profile-use=[use PGO profile for rustc build]:PROFILE:_files' \
'--llvm-profile-use=[use PGO profile for LLVM build]:PROFILE:_files' \
'*--reproducible-artifact=[Additional reproducible artifacts that should be added to the reproducible artifacts archive]:REPRODUCIBLE_ARTIFACT: ' \
'*--set=[override options in config.toml]:section.option=value:( )' \
'*-v[use verbose output (-vv for very verbose)]' \
'*--verbose[use verbose output (-vv for very verbose)]' \
'-i[use incremental compilation]' \
'--incremental[use incremental compilation]' \
'--include-default-paths[include default paths in addition to the provided ones]' \
'--dry-run[dry run; don'\''t build anything]' \
'--dump-bootstrap-shims[Indicates whether to dump the work done from bootstrap shims]' \
'--json-output[use message-format=json]' \
'--bypass-bootstrap-lock[Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)]' \
'--llvm-profile-generate[generate PGO profile with llvm built for rustc]' \
'--enable-bolt-settings[Enable BOLT link flags]' \
'--skip-stage0-validation[Skip stage0 compiler validation]' \
'-h[Print help (see more with '\''--help'\'')]' \
'--help[Print help (see more with '\''--help'\'')]' \
'*::paths -- paths for the subcommand:_files' \
&& ret=0
;;
(clean)
_arguments "${_arguments_options[@]}" : \
'--stage=[Clean a specific stage without touching other artifacts. By default, every stage is cleaned if this option is not used]:N: ' \
'--config=[TOML configuration file for build]:FILE:_files' \
'--build-dir=[Build directory, overrides \`build.build-dir\` in \`config.toml\`]:DIR:_files -/' \
'--build=[build target of the stage0 compiler]:BUILD:( )' \
'--host=[host targets to build]:HOST:( )' \
'--target=[target targets to build]:TARGET:( )' \
'*--exclude=[build paths to exclude]:PATH:_files' \
'*--skip=[build paths to skip]:PATH:_files' \
'--rustc-error-format=[]:RUSTC_ERROR_FORMAT:( )' \
'--on-fail=[command to run on failure]:CMD:_cmdstring' \
'*--keep-stage=[stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'*--keep-stage-std=[stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'--src=[path to the root of the rust checkout]:DIR:_files -/' \
'-j+[number of jobs to run in parallel]:JOBS:( )' \
'--jobs=[number of jobs to run in parallel]:JOBS:( )' \
'--warnings=[if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour]:deny|warn:(deny warn default)' \
'--error-format=[rustc error format]:FORMAT:( )' \
'--color=[whether to use color in cargo and rustc output]:STYLE:(always never auto)' \
'--rust-profile-generate=[generate PGO profile with rustc build]:PROFILE:_files' \
'--rust-profile-use=[use PGO profile for rustc build]:PROFILE:_files' \
'--llvm-profile-use=[use PGO profile for LLVM build]:PROFILE:_files' \
'*--reproducible-artifact=[Additional reproducible artifacts that should be added to the reproducible artifacts archive]:REPRODUCIBLE_ARTIFACT: ' \
'*--set=[override options in config.toml]:section.option=value:( )' \
'--all[Clean the entire build directory (not used by default)]' \
'*-v[use verbose output (-vv for very verbose)]' \
'*--verbose[use verbose output (-vv for very verbose)]' \
'-i[use incremental compilation]' \
'--incremental[use incremental compilation]' \
'--include-default-paths[include default paths in addition to the provided ones]' \
'--dry-run[dry run; don'\''t build anything]' \
'--dump-bootstrap-shims[Indicates whether to dump the work done from bootstrap shims]' \
'--json-output[use message-format=json]' \
'--bypass-bootstrap-lock[Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)]' \
'--llvm-profile-generate[generate PGO profile with llvm built for rustc]' \
'--enable-bolt-settings[Enable BOLT link flags]' \
'--skip-stage0-validation[Skip stage0 compiler validation]' \
'-h[Print help (see more with '\''--help'\'')]' \
'--help[Print help (see more with '\''--help'\'')]' \
'*::paths -- paths for the subcommand:_files' \
&& ret=0
;;
(dist)
_arguments "${_arguments_options[@]}" : \
'--config=[TOML configuration file for build]:FILE:_files' \
'--build-dir=[Build directory, overrides \`build.build-dir\` in \`config.toml\`]:DIR:_files -/' \
'--build=[build target of the stage0 compiler]:BUILD:( )' \
'--host=[host targets to build]:HOST:( )' \
'--target=[target targets to build]:TARGET:( )' \
'*--exclude=[build paths to exclude]:PATH:_files' \
'*--skip=[build paths to skip]:PATH:_files' \
'--rustc-error-format=[]:RUSTC_ERROR_FORMAT:( )' \
'--on-fail=[command to run on failure]:CMD:_cmdstring' \
'--stage=[stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)]:N:( )' \
'*--keep-stage=[stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'*--keep-stage-std=[stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'--src=[path to the root of the rust checkout]:DIR:_files -/' \
'-j+[number of jobs to run in parallel]:JOBS:( )' \
'--jobs=[number of jobs to run in parallel]:JOBS:( )' \
'--warnings=[if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour]:deny|warn:(deny warn default)' \
'--error-format=[rustc error format]:FORMAT:( )' \
'--color=[whether to use color in cargo and rustc output]:STYLE:(always never auto)' \
'--rust-profile-generate=[generate PGO profile with rustc build]:PROFILE:_files' \
'--rust-profile-use=[use PGO profile for rustc build]:PROFILE:_files' \
'--llvm-profile-use=[use PGO profile for LLVM build]:PROFILE:_files' \
'*--reproducible-artifact=[Additional reproducible artifacts that should be added to the reproducible artifacts archive]:REPRODUCIBLE_ARTIFACT: ' \
'*--set=[override options in config.toml]:section.option=value:( )' \
'*-v[use verbose output (-vv for very verbose)]' \
'*--verbose[use verbose output (-vv for very verbose)]' \
'-i[use incremental compilation]' \
'--incremental[use incremental compilation]' \
'--include-default-paths[include default paths in addition to the provided ones]' \
'--dry-run[dry run; don'\''t build anything]' \
'--dump-bootstrap-shims[Indicates whether to dump the work done from bootstrap shims]' \
'--json-output[use message-format=json]' \
'--bypass-bootstrap-lock[Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)]' \
'--llvm-profile-generate[generate PGO profile with llvm built for rustc]' \
'--enable-bolt-settings[Enable BOLT link flags]' \
'--skip-stage0-validation[Skip stage0 compiler validation]' \
'-h[Print help (see more with '\''--help'\'')]' \
'--help[Print help (see more with '\''--help'\'')]' \
'*::paths -- paths for the subcommand:_files' \
&& ret=0
;;
(install)
_arguments "${_arguments_options[@]}" : \
'--config=[TOML configuration file for build]:FILE:_files' \
'--build-dir=[Build directory, overrides \`build.build-dir\` in \`config.toml\`]:DIR:_files -/' \
'--build=[build target of the stage0 compiler]:BUILD:( )' \
'--host=[host targets to build]:HOST:( )' \
'--target=[target targets to build]:TARGET:( )' \
'*--exclude=[build paths to exclude]:PATH:_files' \
'*--skip=[build paths to skip]:PATH:_files' \
'--rustc-error-format=[]:RUSTC_ERROR_FORMAT:( )' \
'--on-fail=[command to run on failure]:CMD:_cmdstring' \
'--stage=[stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)]:N:( )' \
'*--keep-stage=[stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'*--keep-stage-std=[stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'--src=[path to the root of the rust checkout]:DIR:_files -/' \
'-j+[number of jobs to run in parallel]:JOBS:( )' \
'--jobs=[number of jobs to run in parallel]:JOBS:( )' \
'--warnings=[if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour]:deny|warn:(deny warn default)' \
'--error-format=[rustc error format]:FORMAT:( )' \
'--color=[whether to use color in cargo and rustc output]:STYLE:(always never auto)' \
'--rust-profile-generate=[generate PGO profile with rustc build]:PROFILE:_files' \
'--rust-profile-use=[use PGO profile for rustc build]:PROFILE:_files' \
'--llvm-profile-use=[use PGO profile for LLVM build]:PROFILE:_files' \
'*--reproducible-artifact=[Additional reproducible artifacts that should be added to the reproducible artifacts archive]:REPRODUCIBLE_ARTIFACT: ' \
'*--set=[override options in config.toml]:section.option=value:( )' \
'*-v[use verbose output (-vv for very verbose)]' \
'*--verbose[use verbose output (-vv for very verbose)]' \
'-i[use incremental compilation]' \
'--incremental[use incremental compilation]' \
'--include-default-paths[include default paths in addition to the provided ones]' \
'--dry-run[dry run; don'\''t build anything]' \
'--dump-bootstrap-shims[Indicates whether to dump the work done from bootstrap shims]' \
'--json-output[use message-format=json]' \
'--bypass-bootstrap-lock[Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)]' \
'--llvm-profile-generate[generate PGO profile with llvm built for rustc]' \
'--enable-bolt-settings[Enable BOLT link flags]' \
'--skip-stage0-validation[Skip stage0 compiler validation]' \
'-h[Print help (see more with '\''--help'\'')]' \
'--help[Print help (see more with '\''--help'\'')]' \
'*::paths -- paths for the subcommand:_files' \
&& ret=0
;;
(run)
_arguments "${_arguments_options[@]}" : \
'*--args=[arguments for the tool]:ARGS: ' \
'--config=[TOML configuration file for build]:FILE:_files' \
'--build-dir=[Build directory, overrides \`build.build-dir\` in \`config.toml\`]:DIR:_files -/' \
'--build=[build target of the stage0 compiler]:BUILD:( )' \
'--host=[host targets to build]:HOST:( )' \
'--target=[target targets to build]:TARGET:( )' \
'*--exclude=[build paths to exclude]:PATH:_files' \
'*--skip=[build paths to skip]:PATH:_files' \
'--rustc-error-format=[]:RUSTC_ERROR_FORMAT:( )' \
'--on-fail=[command to run on failure]:CMD:_cmdstring' \
'--stage=[stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)]:N:( )' \
'*--keep-stage=[stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'*--keep-stage-std=[stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'--src=[path to the root of the rust checkout]:DIR:_files -/' \
'-j+[number of jobs to run in parallel]:JOBS:( )' \
'--jobs=[number of jobs to run in parallel]:JOBS:( )' \
'--warnings=[if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour]:deny|warn:(deny warn default)' \
'--error-format=[rustc error format]:FORMAT:( )' \
'--color=[whether to use color in cargo and rustc output]:STYLE:(always never auto)' \
'--rust-profile-generate=[generate PGO profile with rustc build]:PROFILE:_files' \
'--rust-profile-use=[use PGO profile for rustc build]:PROFILE:_files' \
'--llvm-profile-use=[use PGO profile for LLVM build]:PROFILE:_files' \
'*--reproducible-artifact=[Additional reproducible artifacts that should be added to the reproducible artifacts archive]:REPRODUCIBLE_ARTIFACT: ' \
'*--set=[override options in config.toml]:section.option=value:( )' \
'*-v[use verbose output (-vv for very verbose)]' \
'*--verbose[use verbose output (-vv for very verbose)]' \
'-i[use incremental compilation]' \
'--incremental[use incremental compilation]' \
'--include-default-paths[include default paths in addition to the provided ones]' \
'--dry-run[dry run; don'\''t build anything]' \
'--dump-bootstrap-shims[Indicates whether to dump the work done from bootstrap shims]' \
'--json-output[use message-format=json]' \
'--bypass-bootstrap-lock[Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)]' \
'--llvm-profile-generate[generate PGO profile with llvm built for rustc]' \
'--enable-bolt-settings[Enable BOLT link flags]' \
'--skip-stage0-validation[Skip stage0 compiler validation]' \
'-h[Print help (see more with '\''--help'\'')]' \
'--help[Print help (see more with '\''--help'\'')]' \
'*::paths -- paths for the subcommand:_files' \
&& ret=0
;;
(setup)
_arguments "${_arguments_options[@]}" : \
'--config=[TOML configuration file for build]:FILE:_files' \
'--build-dir=[Build directory, overrides \`build.build-dir\` in \`config.toml\`]:DIR:_files -/' \
'--build=[build target of the stage0 compiler]:BUILD:( )' \
'--host=[host targets to build]:HOST:( )' \
'--target=[target targets to build]:TARGET:( )' \
'*--exclude=[build paths to exclude]:PATH:_files' \
'*--skip=[build paths to skip]:PATH:_files' \
'--rustc-error-format=[]:RUSTC_ERROR_FORMAT:( )' \
'--on-fail=[command to run on failure]:CMD:_cmdstring' \
'--stage=[stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)]:N:( )' \
'*--keep-stage=[stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'*--keep-stage-std=[stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'--src=[path to the root of the rust checkout]:DIR:_files -/' \
'-j+[number of jobs to run in parallel]:JOBS:( )' \
'--jobs=[number of jobs to run in parallel]:JOBS:( )' \
'--warnings=[if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour]:deny|warn:(deny warn default)' \
'--error-format=[rustc error format]:FORMAT:( )' \
'--color=[whether to use color in cargo and rustc output]:STYLE:(always never auto)' \
'--rust-profile-generate=[generate PGO profile with rustc build]:PROFILE:_files' \
'--rust-profile-use=[use PGO profile for rustc build]:PROFILE:_files' \
'--llvm-profile-use=[use PGO profile for LLVM build]:PROFILE:_files' \
'*--reproducible-artifact=[Additional reproducible artifacts that should be added to the reproducible artifacts archive]:REPRODUCIBLE_ARTIFACT: ' \
'*--set=[override options in config.toml]:section.option=value:( )' \
'*-v[use verbose output (-vv for very verbose)]' \
'*--verbose[use verbose output (-vv for very verbose)]' \
'-i[use incremental compilation]' \
'--incremental[use incremental compilation]' \
'--include-default-paths[include default paths in addition to the provided ones]' \
'--dry-run[dry run; don'\''t build anything]' \
'--dump-bootstrap-shims[Indicates whether to dump the work done from bootstrap shims]' \
'--json-output[use message-format=json]' \
'--bypass-bootstrap-lock[Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)]' \
'--llvm-profile-generate[generate PGO profile with llvm built for rustc]' \
'--enable-bolt-settings[Enable BOLT link flags]' \
'--skip-stage0-validation[Skip stage0 compiler validation]' \
'-h[Print help (see more with '\''--help'\'')]' \
'--help[Print help (see more with '\''--help'\'')]' \
'::profile -- Either the profile for `config.toml` or another setup action. May be omitted to set up interactively:_files' \
'*::paths -- paths for the subcommand:_files' \
&& ret=0
;;
(suggest)
_arguments "${_arguments_options[@]}" : \
'--config=[TOML configuration file for build]:FILE:_files' \
'--build-dir=[Build directory, overrides \`build.build-dir\` in \`config.toml\`]:DIR:_files -/' \
'--build=[build target of the stage0 compiler]:BUILD:( )' \
'--host=[host targets to build]:HOST:( )' \
'--target=[target targets to build]:TARGET:( )' \
'*--exclude=[build paths to exclude]:PATH:_files' \
'*--skip=[build paths to skip]:PATH:_files' \
'--rustc-error-format=[]:RUSTC_ERROR_FORMAT:( )' \
'--on-fail=[command to run on failure]:CMD:_cmdstring' \
'--stage=[stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)]:N:( )' \
'*--keep-stage=[stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'*--keep-stage-std=[stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'--src=[path to the root of the rust checkout]:DIR:_files -/' \
'-j+[number of jobs to run in parallel]:JOBS:( )' \
'--jobs=[number of jobs to run in parallel]:JOBS:( )' \
'--warnings=[if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour]:deny|warn:(deny warn default)' \
'--error-format=[rustc error format]:FORMAT:( )' \
'--color=[whether to use color in cargo and rustc output]:STYLE:(always never auto)' \
'--rust-profile-generate=[generate PGO profile with rustc build]:PROFILE:_files' \
'--rust-profile-use=[use PGO profile for rustc build]:PROFILE:_files' \
'--llvm-profile-use=[use PGO profile for LLVM build]:PROFILE:_files' \
'*--reproducible-artifact=[Additional reproducible artifacts that should be added to the reproducible artifacts archive]:REPRODUCIBLE_ARTIFACT: ' \
'*--set=[override options in config.toml]:section.option=value:( )' \
'--run[run suggested tests]' \
'*-v[use verbose output (-vv for very verbose)]' \
'*--verbose[use verbose output (-vv for very verbose)]' \
'-i[use incremental compilation]' \
'--incremental[use incremental compilation]' \
'--include-default-paths[include default paths in addition to the provided ones]' \
'--dry-run[dry run; don'\''t build anything]' \
'--dump-bootstrap-shims[Indicates whether to dump the work done from bootstrap shims]' \
'--json-output[use message-format=json]' \
'--bypass-bootstrap-lock[Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)]' \
'--llvm-profile-generate[generate PGO profile with llvm built for rustc]' \
'--enable-bolt-settings[Enable BOLT link flags]' \
'--skip-stage0-validation[Skip stage0 compiler validation]' \
'-h[Print help (see more with '\''--help'\'')]' \
'--help[Print help (see more with '\''--help'\'')]' \
'*::paths -- paths for the subcommand:_files' \
&& ret=0
;;
(vendor)
_arguments "${_arguments_options[@]}" : \
'*--sync=[Additional \`Cargo.toml\` to sync and vendor]:SYNC:_files' \
'--config=[TOML configuration file for build]:FILE:_files' \
'--build-dir=[Build directory, overrides \`build.build-dir\` in \`config.toml\`]:DIR:_files -/' \
'--build=[build target of the stage0 compiler]:BUILD:( )' \
'--host=[host targets to build]:HOST:( )' \
'--target=[target targets to build]:TARGET:( )' \
'*--exclude=[build paths to exclude]:PATH:_files' \
'*--skip=[build paths to skip]:PATH:_files' \
'--rustc-error-format=[]:RUSTC_ERROR_FORMAT:( )' \
'--on-fail=[command to run on failure]:CMD:_cmdstring' \
'--stage=[stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)]:N:( )' \
'*--keep-stage=[stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'*--keep-stage-std=[stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'--src=[path to the root of the rust checkout]:DIR:_files -/' \
'-j+[number of jobs to run in parallel]:JOBS:( )' \
'--jobs=[number of jobs to run in parallel]:JOBS:( )' \
'--warnings=[if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour]:deny|warn:(deny warn default)' \
'--error-format=[rustc error format]:FORMAT:( )' \
'--color=[whether to use color in cargo and rustc output]:STYLE:(always never auto)' \
'--rust-profile-generate=[generate PGO profile with rustc build]:PROFILE:_files' \
'--rust-profile-use=[use PGO profile for rustc build]:PROFILE:_files' \
'--llvm-profile-use=[use PGO profile for LLVM build]:PROFILE:_files' \
'*--reproducible-artifact=[Additional reproducible artifacts that should be added to the reproducible artifacts archive]:REPRODUCIBLE_ARTIFACT: ' \
'*--set=[override options in config.toml]:section.option=value:( )' \
'--versioned-dirs[Always include version in subdir name]' \
'*-v[use verbose output (-vv for very verbose)]' \
'*--verbose[use verbose output (-vv for very verbose)]' \
'-i[use incremental compilation]' \
'--incremental[use incremental compilation]' \
'--include-default-paths[include default paths in addition to the provided ones]' \
'--dry-run[dry run; don'\''t build anything]' \
'--dump-bootstrap-shims[Indicates whether to dump the work done from bootstrap shims]' \
'--json-output[use message-format=json]' \
'--bypass-bootstrap-lock[Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)]' \
'--llvm-profile-generate[generate PGO profile with llvm built for rustc]' \
'--enable-bolt-settings[Enable BOLT link flags]' \
'--skip-stage0-validation[Skip stage0 compiler validation]' \
'-h[Print help (see more with '\''--help'\'')]' \
'--help[Print help (see more with '\''--help'\'')]' \
'*::paths -- paths for the subcommand:_files' \
&& ret=0
;;
(perf)
_arguments "${_arguments_options[@]}" : \
'--config=[TOML configuration file for build]:FILE:_files' \
'--build-dir=[Build directory, overrides \`build.build-dir\` in \`config.toml\`]:DIR:_files -/' \
'--build=[build target of the stage0 compiler]:BUILD:( )' \
'--host=[host targets to build]:HOST:( )' \
'--target=[target targets to build]:TARGET:( )' \
'*--exclude=[build paths to exclude]:PATH:_files' \
'*--skip=[build paths to skip]:PATH:_files' \
'--rustc-error-format=[]:RUSTC_ERROR_FORMAT:( )' \
'--on-fail=[command to run on failure]:CMD:_cmdstring' \
'--stage=[stage to build (indicates compiler to use/test, e.g., stage 0 uses the bootstrap compiler, stage 1 the stage 0 rustc artifacts, etc.)]:N:( )' \
'*--keep-stage=[stage(s) to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'*--keep-stage-std=[stage(s) of the standard library to keep without recompiling (pass multiple times to keep e.g., both stages 0 and 1)]:N:( )' \
'--src=[path to the root of the rust checkout]:DIR:_files -/' \
'-j+[number of jobs to run in parallel]:JOBS:( )' \
'--jobs=[number of jobs to run in parallel]:JOBS:( )' \
'--warnings=[if value is deny, will deny warnings if value is warn, will emit warnings otherwise, use the default configured behaviour]:deny|warn:(deny warn default)' \
'--error-format=[rustc error format]:FORMAT:( )' \
'--color=[whether to use color in cargo and rustc output]:STYLE:(always never auto)' \
'--rust-profile-generate=[generate PGO profile with rustc build]:PROFILE:_files' \
'--rust-profile-use=[use PGO profile for rustc build]:PROFILE:_files' \
'--llvm-profile-use=[use PGO profile for LLVM build]:PROFILE:_files' \
'*--reproducible-artifact=[Additional reproducible artifacts that should be added to the reproducible artifacts archive]:REPRODUCIBLE_ARTIFACT: ' \
'*--set=[override options in config.toml]:section.option=value:( )' \
'*-v[use verbose output (-vv for very verbose)]' \
'*--verbose[use verbose output (-vv for very verbose)]' \
'-i[use incremental compilation]' \
'--incremental[use incremental compilation]' \
'--include-default-paths[include default paths in addition to the provided ones]' \
'--dry-run[dry run; don'\''t build anything]' \
'--dump-bootstrap-shims[Indicates whether to dump the work done from bootstrap shims]' \
'--json-output[use message-format=json]' \
'--bypass-bootstrap-lock[Bootstrap uses this value to decide whether it should bypass locking the build process. This is rarely needed (e.g., compiling the std library for different targets in parallel)]' \
'--llvm-profile-generate[generate PGO profile with llvm built for rustc]' \
'--enable-bolt-settings[Enable BOLT link flags]' \
'--skip-stage0-validation[Skip stage0 compiler validation]' \
'-h[Print help (see more with '\''--help'\'')]' \
'--help[Print help (see more with '\''--help'\'')]' \
'*::paths -- paths for the subcommand:_files' \
&& ret=0
;;
        esac
    ;;
esac
}

(( $+functions[_x.py_commands] )) ||
_x.py_commands() {
    local commands; commands=(
'build:Compile either the compiler or libraries' \
'check:Compile either the compiler or libraries, using cargo check' \
'clippy:Run Clippy (uses rustup/cargo-installed clippy binary)' \
'fix:Run cargo fix' \
'fmt:Run rustfmt' \
'doc:Build documentation' \
'test:Build and run some test suites' \
'miri:Build and run some test suites *in Miri*' \
'bench:Build and run some benchmarks' \
'clean:Clean out build directories' \
'dist:Build distribution artifacts' \
'install:Install distribution artifacts' \
'run:Run tools contained in this repository' \
'setup:Set up the environment for development' \
'suggest:Suggest a subset of tests to run, based on modified files' \
'vendor:Vendor dependencies' \
'perf:Perform profiling and benchmarking of the compiler using the \`rustc-perf-wrapper\` tool' \
    )
    _describe -t commands 'x.py commands' commands "$@"
}
(( $+functions[_x.py__bench_commands] )) ||
_x.py__bench_commands() {
    local commands; commands=()
    _describe -t commands 'x.py bench commands' commands "$@"
}
(( $+functions[_x.py__build_commands] )) ||
_x.py__build_commands() {
    local commands; commands=()
    _describe -t commands 'x.py build commands' commands "$@"
}
(( $+functions[_x.py__check_commands] )) ||
_x.py__check_commands() {
    local commands; commands=()
    _describe -t commands 'x.py check commands' commands "$@"
}
(( $+functions[_x.py__clean_commands] )) ||
_x.py__clean_commands() {
    local commands; commands=()
    _describe -t commands 'x.py clean commands' commands "$@"
}
(( $+functions[_x.py__clippy_commands] )) ||
_x.py__clippy_commands() {
    local commands; commands=()
    _describe -t commands 'x.py clippy commands' commands "$@"
}
(( $+functions[_x.py__dist_commands] )) ||
_x.py__dist_commands() {
    local commands; commands=()
    _describe -t commands 'x.py dist commands' commands "$@"
}
(( $+functions[_x.py__doc_commands] )) ||
_x.py__doc_commands() {
    local commands; commands=()
    _describe -t commands 'x.py doc commands' commands "$@"
}
(( $+functions[_x.py__fix_commands] )) ||
_x.py__fix_commands() {
    local commands; commands=()
    _describe -t commands 'x.py fix commands' commands "$@"
}
(( $+functions[_x.py__fmt_commands] )) ||
_x.py__fmt_commands() {
    local commands; commands=()
    _describe -t commands 'x.py fmt commands' commands "$@"
}
(( $+functions[_x.py__install_commands] )) ||
_x.py__install_commands() {
    local commands; commands=()
    _describe -t commands 'x.py install commands' commands "$@"
}
(( $+functions[_x.py__miri_commands] )) ||
_x.py__miri_commands() {
    local commands; commands=()
    _describe -t commands 'x.py miri commands' commands "$@"
}
(( $+functions[_x.py__perf_commands] )) ||
_x.py__perf_commands() {
    local commands; commands=()
    _describe -t commands 'x.py perf commands' commands "$@"
}
(( $+functions[_x.py__run_commands] )) ||
_x.py__run_commands() {
    local commands; commands=()
    _describe -t commands 'x.py run commands' commands "$@"
}
(( $+functions[_x.py__setup_commands] )) ||
_x.py__setup_commands() {
    local commands; commands=()
    _describe -t commands 'x.py setup commands' commands "$@"
}
(( $+functions[_x.py__suggest_commands] )) ||
_x.py__suggest_commands() {
    local commands; commands=()
    _describe -t commands 'x.py suggest commands' commands "$@"
}
(( $+functions[_x.py__test_commands] )) ||
_x.py__test_commands() {
    local commands; commands=()
    _describe -t commands 'x.py test commands' commands "$@"
}
(( $+functions[_x.py__vendor_commands] )) ||
_x.py__vendor_commands() {
    local commands; commands=()
    _describe -t commands 'x.py vendor commands' commands "$@"
}

if [ "$funcstack[1]" = "_x.py" ]; then
    _x.py "$@"
else
    compdef _x.py x.py
fi
