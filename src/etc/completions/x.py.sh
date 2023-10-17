_x.py() {
    local i cur prev opts cmd
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    cmd=""
    opts=""

    for i in ${COMP_WORDS[@]}
    do
        case "${cmd},${i}" in
            ",$1")
                cmd="x.py"
                ;;
            bootstrap,bench)
                cmd="bootstrap__bench"
                ;;
            bootstrap,build)
                cmd="bootstrap__build"
                ;;
            bootstrap,check)
                cmd="bootstrap__check"
                ;;
            bootstrap,clean)
                cmd="bootstrap__clean"
                ;;
            bootstrap,clippy)
                cmd="bootstrap__clippy"
                ;;
            bootstrap,dist)
                cmd="bootstrap__dist"
                ;;
            bootstrap,doc)
                cmd="bootstrap__doc"
                ;;
            bootstrap,fix)
                cmd="bootstrap__fix"
                ;;
            bootstrap,fmt)
                cmd="bootstrap__fmt"
                ;;
            bootstrap,install)
                cmd="bootstrap__install"
                ;;
            bootstrap,run)
                cmd="bootstrap__run"
                ;;
            bootstrap,setup)
                cmd="bootstrap__setup"
                ;;
            bootstrap,suggest)
                cmd="bootstrap__suggest"
                ;;
            bootstrap,test)
                cmd="bootstrap__test"
                ;;
            *)
                ;;
        esac
    done

    case "${cmd}" in
        x.py)
            opts="-v -i -j -h --verbose --incremental --config --build-dir --build --host --target --exclude --skip --include-default-paths --rustc-error-format --on-fail --dry-run --stage --keep-stage --keep-stage-std --src --jobs --warnings --error-format --json-output --color --llvm-skip-rebuild --rust-profile-generate --rust-profile-use --llvm-profile-use --llvm-profile-generate --enable-bolt-settings --reproducible-artifact --set --help [PATHS]... [ARGS]... build check clippy fix fmt doc test bench clean dist install run setup suggest"
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 1 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --config)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build-dir)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --host)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --target)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --exclude)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --skip)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rustc-error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --on-fail)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage-std)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --src)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --jobs)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                -j)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --warnings)
                    COMPREPLY=($(compgen -W "deny warn default" -- "${cur}"))
                    return 0
                    ;;
                --error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --color)
                    COMPREPLY=($(compgen -W "always never auto" -- "${cur}"))
                    return 0
                    ;;
                --llvm-skip-rebuild)
                    COMPREPLY=($(compgen -W "true false" -- "${cur}"))
                    return 0
                    ;;
                --rust-profile-generate)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rust-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --llvm-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --reproducible-artifact)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --set)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        x.py__bench)
            opts="-v -i -j -h --test-args --verbose --incremental --config --build-dir --build --host --target --exclude --skip --include-default-paths --rustc-error-format --on-fail --dry-run --stage --keep-stage --keep-stage-std --src --jobs --warnings --error-format --json-output --color --llvm-skip-rebuild --rust-profile-generate --rust-profile-use --llvm-profile-use --llvm-profile-generate --enable-bolt-settings --reproducible-artifact --set --help [PATHS]... [ARGS]..."
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 2 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --test-args)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --config)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build-dir)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --host)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --target)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --exclude)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --skip)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rustc-error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --on-fail)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage-std)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --src)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --jobs)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                -j)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --warnings)
                    COMPREPLY=($(compgen -W "deny warn default" -- "${cur}"))
                    return 0
                    ;;
                --error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --color)
                    COMPREPLY=($(compgen -W "always never auto" -- "${cur}"))
                    return 0
                    ;;
                --llvm-skip-rebuild)
                    COMPREPLY=($(compgen -W "true false" -- "${cur}"))
                    return 0
                    ;;
                --rust-profile-generate)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rust-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --llvm-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --reproducible-artifact)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --set)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        x.py__build)
            opts="-v -i -j -h --verbose --incremental --config --build-dir --build --host --target --exclude --skip --include-default-paths --rustc-error-format --on-fail --dry-run --stage --keep-stage --keep-stage-std --src --jobs --warnings --error-format --json-output --color --llvm-skip-rebuild --rust-profile-generate --rust-profile-use --llvm-profile-use --llvm-profile-generate --enable-bolt-settings --reproducible-artifact --set --help [PATHS]... [ARGS]..."
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 2 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --config)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build-dir)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --host)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --target)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --exclude)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --skip)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rustc-error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --on-fail)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage-std)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --src)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --jobs)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                -j)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --warnings)
                    COMPREPLY=($(compgen -W "deny warn default" -- "${cur}"))
                    return 0
                    ;;
                --error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --color)
                    COMPREPLY=($(compgen -W "always never auto" -- "${cur}"))
                    return 0
                    ;;
                --llvm-skip-rebuild)
                    COMPREPLY=($(compgen -W "true false" -- "${cur}"))
                    return 0
                    ;;
                --rust-profile-generate)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rust-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --llvm-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --reproducible-artifact)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --set)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        x.py__check)
            opts="-v -i -j -h --all-targets --verbose --incremental --config --build-dir --build --host --target --exclude --skip --include-default-paths --rustc-error-format --on-fail --dry-run --stage --keep-stage --keep-stage-std --src --jobs --warnings --error-format --json-output --color --llvm-skip-rebuild --rust-profile-generate --rust-profile-use --llvm-profile-use --llvm-profile-generate --enable-bolt-settings --reproducible-artifact --set --help [PATHS]... [ARGS]..."
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 2 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --config)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build-dir)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --host)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --target)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --exclude)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --skip)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rustc-error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --on-fail)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage-std)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --src)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --jobs)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                -j)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --warnings)
                    COMPREPLY=($(compgen -W "deny warn default" -- "${cur}"))
                    return 0
                    ;;
                --error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --color)
                    COMPREPLY=($(compgen -W "always never auto" -- "${cur}"))
                    return 0
                    ;;
                --llvm-skip-rebuild)
                    COMPREPLY=($(compgen -W "true false" -- "${cur}"))
                    return 0
                    ;;
                --rust-profile-generate)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rust-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --llvm-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --reproducible-artifact)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --set)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        x.py__clean)
            opts="-v -i -j -h --all --stage --verbose --incremental --config --build-dir --build --host --target --exclude --skip --include-default-paths --rustc-error-format --on-fail --dry-run --keep-stage --keep-stage-std --src --jobs --warnings --error-format --json-output --color --llvm-skip-rebuild --rust-profile-generate --rust-profile-use --llvm-profile-use --llvm-profile-generate --enable-bolt-settings --reproducible-artifact --set --help [PATHS]... [ARGS]..."
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 2 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --stage)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --config)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build-dir)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --host)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --target)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --exclude)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --skip)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rustc-error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --on-fail)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --keep-stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage-std)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --src)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --jobs)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                -j)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --warnings)
                    COMPREPLY=($(compgen -W "deny warn default" -- "${cur}"))
                    return 0
                    ;;
                --error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --color)
                    COMPREPLY=($(compgen -W "always never auto" -- "${cur}"))
                    return 0
                    ;;
                --llvm-skip-rebuild)
                    COMPREPLY=($(compgen -W "true false" -- "${cur}"))
                    return 0
                    ;;
                --rust-profile-generate)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rust-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --llvm-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --reproducible-artifact)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --set)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        x.py__clippy)
            opts="-A -D -W -F -v -i -j -h --fix --verbose --incremental --config --build-dir --build --host --target --exclude --skip --include-default-paths --rustc-error-format --on-fail --dry-run --stage --keep-stage --keep-stage-std --src --jobs --warnings --error-format --json-output --color --llvm-skip-rebuild --rust-profile-generate --rust-profile-use --llvm-profile-use --llvm-profile-generate --enable-bolt-settings --reproducible-artifact --set --help [PATHS]... [ARGS]..."
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 2 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                -A)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                -D)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                -W)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                -F)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --config)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build-dir)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --host)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --target)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --exclude)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --skip)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rustc-error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --on-fail)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage-std)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --src)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --jobs)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                -j)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --warnings)
                    COMPREPLY=($(compgen -W "deny warn default" -- "${cur}"))
                    return 0
                    ;;
                --error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --color)
                    COMPREPLY=($(compgen -W "always never auto" -- "${cur}"))
                    return 0
                    ;;
                --llvm-skip-rebuild)
                    COMPREPLY=($(compgen -W "true false" -- "${cur}"))
                    return 0
                    ;;
                --rust-profile-generate)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rust-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --llvm-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --reproducible-artifact)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --set)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        x.py__dist)
            opts="-v -i -j -h --verbose --incremental --config --build-dir --build --host --target --exclude --skip --include-default-paths --rustc-error-format --on-fail --dry-run --stage --keep-stage --keep-stage-std --src --jobs --warnings --error-format --json-output --color --llvm-skip-rebuild --rust-profile-generate --rust-profile-use --llvm-profile-use --llvm-profile-generate --enable-bolt-settings --reproducible-artifact --set --help [PATHS]... [ARGS]..."
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 2 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --config)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build-dir)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --host)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --target)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --exclude)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --skip)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rustc-error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --on-fail)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage-std)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --src)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --jobs)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                -j)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --warnings)
                    COMPREPLY=($(compgen -W "deny warn default" -- "${cur}"))
                    return 0
                    ;;
                --error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --color)
                    COMPREPLY=($(compgen -W "always never auto" -- "${cur}"))
                    return 0
                    ;;
                --llvm-skip-rebuild)
                    COMPREPLY=($(compgen -W "true false" -- "${cur}"))
                    return 0
                    ;;
                --rust-profile-generate)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rust-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --llvm-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --reproducible-artifact)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --set)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        x.py__doc)
            opts="-v -i -j -h --open --json --verbose --incremental --config --build-dir --build --host --target --exclude --skip --include-default-paths --rustc-error-format --on-fail --dry-run --stage --keep-stage --keep-stage-std --src --jobs --warnings --error-format --json-output --color --llvm-skip-rebuild --rust-profile-generate --rust-profile-use --llvm-profile-use --llvm-profile-generate --enable-bolt-settings --reproducible-artifact --set --help [PATHS]... [ARGS]..."
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 2 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --config)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build-dir)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --host)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --target)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --exclude)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --skip)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rustc-error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --on-fail)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage-std)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --src)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --jobs)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                -j)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --warnings)
                    COMPREPLY=($(compgen -W "deny warn default" -- "${cur}"))
                    return 0
                    ;;
                --error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --color)
                    COMPREPLY=($(compgen -W "always never auto" -- "${cur}"))
                    return 0
                    ;;
                --llvm-skip-rebuild)
                    COMPREPLY=($(compgen -W "true false" -- "${cur}"))
                    return 0
                    ;;
                --rust-profile-generate)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rust-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --llvm-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --reproducible-artifact)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --set)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        x.py__fix)
            opts="-v -i -j -h --verbose --incremental --config --build-dir --build --host --target --exclude --skip --include-default-paths --rustc-error-format --on-fail --dry-run --stage --keep-stage --keep-stage-std --src --jobs --warnings --error-format --json-output --color --llvm-skip-rebuild --rust-profile-generate --rust-profile-use --llvm-profile-use --llvm-profile-generate --enable-bolt-settings --reproducible-artifact --set --help [PATHS]... [ARGS]..."
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 2 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --config)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build-dir)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --host)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --target)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --exclude)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --skip)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rustc-error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --on-fail)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage-std)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --src)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --jobs)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                -j)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --warnings)
                    COMPREPLY=($(compgen -W "deny warn default" -- "${cur}"))
                    return 0
                    ;;
                --error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --color)
                    COMPREPLY=($(compgen -W "always never auto" -- "${cur}"))
                    return 0
                    ;;
                --llvm-skip-rebuild)
                    COMPREPLY=($(compgen -W "true false" -- "${cur}"))
                    return 0
                    ;;
                --rust-profile-generate)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rust-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --llvm-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --reproducible-artifact)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --set)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        x.py__fmt)
            opts="-v -i -j -h --check --verbose --incremental --config --build-dir --build --host --target --exclude --skip --include-default-paths --rustc-error-format --on-fail --dry-run --stage --keep-stage --keep-stage-std --src --jobs --warnings --error-format --json-output --color --llvm-skip-rebuild --rust-profile-generate --rust-profile-use --llvm-profile-use --llvm-profile-generate --enable-bolt-settings --reproducible-artifact --set --help [PATHS]... [ARGS]..."
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 2 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --config)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build-dir)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --host)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --target)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --exclude)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --skip)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rustc-error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --on-fail)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage-std)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --src)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --jobs)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                -j)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --warnings)
                    COMPREPLY=($(compgen -W "deny warn default" -- "${cur}"))
                    return 0
                    ;;
                --error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --color)
                    COMPREPLY=($(compgen -W "always never auto" -- "${cur}"))
                    return 0
                    ;;
                --llvm-skip-rebuild)
                    COMPREPLY=($(compgen -W "true false" -- "${cur}"))
                    return 0
                    ;;
                --rust-profile-generate)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rust-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --llvm-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --reproducible-artifact)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --set)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        x.py__install)
            opts="-v -i -j -h --verbose --incremental --config --build-dir --build --host --target --exclude --skip --include-default-paths --rustc-error-format --on-fail --dry-run --stage --keep-stage --keep-stage-std --src --jobs --warnings --error-format --json-output --color --llvm-skip-rebuild --rust-profile-generate --rust-profile-use --llvm-profile-use --llvm-profile-generate --enable-bolt-settings --reproducible-artifact --set --help [PATHS]... [ARGS]..."
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 2 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --config)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build-dir)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --host)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --target)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --exclude)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --skip)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rustc-error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --on-fail)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage-std)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --src)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --jobs)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                -j)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --warnings)
                    COMPREPLY=($(compgen -W "deny warn default" -- "${cur}"))
                    return 0
                    ;;
                --error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --color)
                    COMPREPLY=($(compgen -W "always never auto" -- "${cur}"))
                    return 0
                    ;;
                --llvm-skip-rebuild)
                    COMPREPLY=($(compgen -W "true false" -- "${cur}"))
                    return 0
                    ;;
                --rust-profile-generate)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rust-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --llvm-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --reproducible-artifact)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --set)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        x.py__run)
            opts="-v -i -j -h --args --verbose --incremental --config --build-dir --build --host --target --exclude --skip --include-default-paths --rustc-error-format --on-fail --dry-run --stage --keep-stage --keep-stage-std --src --jobs --warnings --error-format --json-output --color --llvm-skip-rebuild --rust-profile-generate --rust-profile-use --llvm-profile-use --llvm-profile-generate --enable-bolt-settings --reproducible-artifact --set --help [PATHS]... [ARGS]..."
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 2 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --args)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --config)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build-dir)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --host)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --target)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --exclude)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --skip)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rustc-error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --on-fail)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage-std)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --src)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --jobs)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                -j)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --warnings)
                    COMPREPLY=($(compgen -W "deny warn default" -- "${cur}"))
                    return 0
                    ;;
                --error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --color)
                    COMPREPLY=($(compgen -W "always never auto" -- "${cur}"))
                    return 0
                    ;;
                --llvm-skip-rebuild)
                    COMPREPLY=($(compgen -W "true false" -- "${cur}"))
                    return 0
                    ;;
                --rust-profile-generate)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rust-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --llvm-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --reproducible-artifact)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --set)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        x.py__setup)
            opts="-v -i -j -h --verbose --incremental --config --build-dir --build --host --target --exclude --skip --include-default-paths --rustc-error-format --on-fail --dry-run --stage --keep-stage --keep-stage-std --src --jobs --warnings --error-format --json-output --color --llvm-skip-rebuild --rust-profile-generate --rust-profile-use --llvm-profile-use --llvm-profile-generate --enable-bolt-settings --reproducible-artifact --set --help [<PROFILE>|hook|vscode|link] [PATHS]... [ARGS]..."
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 2 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --config)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build-dir)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --host)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --target)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --exclude)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --skip)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rustc-error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --on-fail)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage-std)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --src)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --jobs)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                -j)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --warnings)
                    COMPREPLY=($(compgen -W "deny warn default" -- "${cur}"))
                    return 0
                    ;;
                --error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --color)
                    COMPREPLY=($(compgen -W "always never auto" -- "${cur}"))
                    return 0
                    ;;
                --llvm-skip-rebuild)
                    COMPREPLY=($(compgen -W "true false" -- "${cur}"))
                    return 0
                    ;;
                --rust-profile-generate)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rust-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --llvm-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --reproducible-artifact)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --set)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        x.py__suggest)
            opts="-v -i -j -h --run --verbose --incremental --config --build-dir --build --host --target --exclude --skip --include-default-paths --rustc-error-format --on-fail --dry-run --stage --keep-stage --keep-stage-std --src --jobs --warnings --error-format --json-output --color --llvm-skip-rebuild --rust-profile-generate --rust-profile-use --llvm-profile-use --llvm-profile-generate --enable-bolt-settings --reproducible-artifact --set --help [PATHS]... [ARGS]..."
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 2 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --config)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build-dir)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --host)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --target)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --exclude)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --skip)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rustc-error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --on-fail)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage-std)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --src)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --jobs)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                -j)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --warnings)
                    COMPREPLY=($(compgen -W "deny warn default" -- "${cur}"))
                    return 0
                    ;;
                --error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --color)
                    COMPREPLY=($(compgen -W "always never auto" -- "${cur}"))
                    return 0
                    ;;
                --llvm-skip-rebuild)
                    COMPREPLY=($(compgen -W "true false" -- "${cur}"))
                    return 0
                    ;;
                --rust-profile-generate)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rust-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --llvm-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --reproducible-artifact)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --set)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        x.py__test)
            opts="-v -i -j -h --no-fail-fast --skip --test-args --rustc-args --no-doc --doc --bless --extra-checks --force-rerun --only-modified --compare-mode --pass --run --rustfix-coverage --verbose --incremental --config --build-dir --build --host --target --exclude --include-default-paths --rustc-error-format --on-fail --dry-run --stage --keep-stage --keep-stage-std --src --jobs --warnings --error-format --json-output --color --llvm-skip-rebuild --rust-profile-generate --rust-profile-use --llvm-profile-use --llvm-profile-generate --enable-bolt-settings --reproducible-artifact --set --help [PATHS]... [ARGS]..."
            if [[ ${cur} == -* || ${COMP_CWORD} -eq 2 ]] ; then
                COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
                return 0
            fi
            case "${prev}" in
                --skip)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --test-args)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rustc-args)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --extra-checks)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --compare-mode)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --pass)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --run)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --config)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build-dir)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --build)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --host)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --target)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --exclude)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rustc-error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --on-fail)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --keep-stage-std)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --src)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --jobs)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                -j)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --warnings)
                    COMPREPLY=($(compgen -W "deny warn default" -- "${cur}"))
                    return 0
                    ;;
                --error-format)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                --color)
                    COMPREPLY=($(compgen -W "always never auto" -- "${cur}"))
                    return 0
                    ;;
                --llvm-skip-rebuild)
                    COMPREPLY=($(compgen -W "true false" -- "${cur}"))
                    return 0
                    ;;
                --rust-profile-generate)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --rust-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --llvm-profile-use)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --reproducible-artifact)
                    COMPREPLY=($(compgen -f "${cur}"))
                    return 0
                    ;;
                --set)
                    COMPREPLY=("${cur}")
                    return 0
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
    esac
}

complete -F _x.py -o nosort -o bashdefault -o default x.py
