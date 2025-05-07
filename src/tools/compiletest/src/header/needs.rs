use crate::common::{Config, KNOWN_CRATE_TYPES, KNOWN_TARGET_HAS_ATOMIC_WIDTHS, Sanitizer};
use crate::header::{IgnoreDecision, llvm_has_libzstd};

pub(super) fn handle_needs(
    cache: &CachedNeedsConditions,
    config: &Config,
    ln: &str,
) -> IgnoreDecision {
    // Note that we intentionally still put the needs- prefix here to make the file show up when
    // grepping for a directive name, even though we could technically strip that.
    let needs = &[
        Need {
            name: "needs-asm-support",
            condition: config.has_asm_support(),
            ignore_reason: "ignored on targets without inline assembly support",
        },
        Need {
            name: "needs-sanitizer-support",
            condition: cache.sanitizer_support,
            ignore_reason: "ignored on targets without sanitizers support",
        },
        Need {
            name: "needs-sanitizer-address",
            condition: cache.sanitizer_address,
            ignore_reason: "ignored on targets without address sanitizer",
        },
        Need {
            name: "needs-sanitizer-cfi",
            condition: cache.sanitizer_cfi,
            ignore_reason: "ignored on targets without CFI sanitizer",
        },
        Need {
            name: "needs-sanitizer-dataflow",
            condition: cache.sanitizer_dataflow,
            ignore_reason: "ignored on targets without dataflow sanitizer",
        },
        Need {
            name: "needs-sanitizer-kcfi",
            condition: cache.sanitizer_kcfi,
            ignore_reason: "ignored on targets without kernel CFI sanitizer",
        },
        Need {
            name: "needs-sanitizer-kasan",
            condition: cache.sanitizer_kasan,
            ignore_reason: "ignored on targets without kernel address sanitizer",
        },
        Need {
            name: "needs-sanitizer-leak",
            condition: cache.sanitizer_leak,
            ignore_reason: "ignored on targets without leak sanitizer",
        },
        Need {
            name: "needs-sanitizer-memory",
            condition: cache.sanitizer_memory,
            ignore_reason: "ignored on targets without memory sanitizer",
        },
        Need {
            name: "needs-sanitizer-thread",
            condition: cache.sanitizer_thread,
            ignore_reason: "ignored on targets without thread sanitizer",
        },
        Need {
            name: "needs-sanitizer-hwaddress",
            condition: cache.sanitizer_hwaddress,
            ignore_reason: "ignored on targets without hardware-assisted address sanitizer",
        },
        Need {
            name: "needs-sanitizer-memtag",
            condition: cache.sanitizer_memtag,
            ignore_reason: "ignored on targets without memory tagging sanitizer",
        },
        Need {
            name: "needs-sanitizer-shadow-call-stack",
            condition: cache.sanitizer_shadow_call_stack,
            ignore_reason: "ignored on targets without shadow call stacks",
        },
        Need {
            name: "needs-sanitizer-safestack",
            condition: cache.sanitizer_safestack,
            ignore_reason: "ignored on targets without SafeStack support",
        },
        Need {
            name: "needs-enzyme",
            condition: config.has_enzyme,
            ignore_reason: "ignored when LLVM Enzyme is disabled",
        },
        Need {
            name: "needs-run-enabled",
            condition: config.run_enabled(),
            ignore_reason: "ignored when running the resulting test binaries is disabled",
        },
        Need {
            name: "needs-threads",
            condition: config.has_threads(),
            ignore_reason: "ignored on targets without threading support",
        },
        Need {
            name: "needs-subprocess",
            condition: config.has_subprocess_support(),
            ignore_reason: "ignored on targets without subprocess support",
        },
        Need {
            name: "needs-unwind",
            condition: config.can_unwind(),
            ignore_reason: "ignored on targets without unwinding support",
        },
        Need {
            name: "needs-profiler-runtime",
            condition: config.profiler_runtime,
            ignore_reason: "ignored when the profiler runtime is not available",
        },
        Need {
            name: "needs-force-clang-based-tests",
            condition: config.run_clang_based_tests_with.is_some(),
            ignore_reason: "ignored when RUSTBUILD_FORCE_CLANG_BASED_TESTS is not set",
        },
        Need {
            name: "needs-xray",
            condition: cache.xray,
            ignore_reason: "ignored on targets without xray tracing",
        },
        Need {
            name: "needs-rust-lld",
            condition: cache.rust_lld,
            ignore_reason: "ignored on targets without Rust's LLD",
        },
        Need {
            name: "needs-dlltool",
            condition: cache.dlltool,
            ignore_reason: "ignored when dlltool for the current architecture is not present",
        },
        Need {
            name: "needs-git-hash",
            condition: config.git_hash,
            ignore_reason: "ignored when git hashes have been omitted for building",
        },
        Need {
            name: "needs-dynamic-linking",
            condition: config.target_cfg().dynamic_linking,
            ignore_reason: "ignored on targets without dynamic linking",
        },
        Need {
            name: "needs-relocation-model-pic",
            condition: config.target_cfg().relocation_model == "pic",
            ignore_reason: "ignored on targets without PIC relocation model",
        },
        Need {
            name: "needs-deterministic-layouts",
            condition: !config.rust_randomized_layout,
            ignore_reason: "ignored when randomizing layouts",
        },
        Need {
            name: "needs-wasmtime",
            condition: config.runner.as_ref().is_some_and(|r| r.contains("wasmtime")),
            ignore_reason: "ignored when wasmtime runner is not available",
        },
        Need {
            name: "needs-symlink",
            condition: cache.symlinks,
            ignore_reason: "ignored if symlinks are unavailable",
        },
        Need {
            name: "needs-llvm-zstd",
            condition: cache.llvm_zstd,
            ignore_reason: "ignored if LLVM wasn't build with zstd for ELF section compression",
        },
        Need {
            name: "needs-rustc-debug-assertions",
            condition: config.with_rustc_debug_assertions,
            ignore_reason: "ignored if rustc wasn't built with debug assertions",
        },
        Need {
            name: "needs-std-debug-assertions",
            condition: config.with_std_debug_assertions,
            ignore_reason: "ignored if std wasn't built with debug assertions",
        },
    ];

    let (name, rest) = match ln.split_once([':', ' ']) {
        Some((name, rest)) => (name, Some(rest)),
        None => (ln, None),
    };

    // FIXME(jieyouxu): tighten up this parsing to reject using both `:` and ` ` as means to
    // delineate value.
    if name == "needs-target-has-atomic" {
        let Some(rest) = rest else {
            return IgnoreDecision::Error {
                message: "expected `needs-target-has-atomic` to have a comma-separated list of atomic widths".to_string(),
            };
        };

        // Expect directive value to be a list of comma-separated atomic widths.
        let specified_widths = rest
            .split(',')
            .map(|width| width.trim())
            .map(ToString::to_string)
            .collect::<Vec<String>>();

        for width in &specified_widths {
            if !KNOWN_TARGET_HAS_ATOMIC_WIDTHS.contains(&width.as_str()) {
                return IgnoreDecision::Error {
                    message: format!(
                        "unknown width specified in `needs-target-has-atomic`: `{width}` is not a \
                        known `target_has_atomic_width`, known values are `{:?}`",
                        KNOWN_TARGET_HAS_ATOMIC_WIDTHS
                    ),
                };
            }
        }

        let satisfies_all_specified_widths = specified_widths
            .iter()
            .all(|specified| config.target_cfg().target_has_atomic.contains(specified));
        if satisfies_all_specified_widths {
            return IgnoreDecision::Continue;
        } else {
            return IgnoreDecision::Ignore {
                reason: format!(
                    "skipping test as target does not support all of the required `target_has_atomic` widths `{:?}`",
                    specified_widths
                ),
            };
        }
    }

    // FIXME(jieyouxu): share multi-value directive logic with `needs-target-has-atomic` above.
    if name == "needs-crate-type" {
        let Some(rest) = rest else {
            return IgnoreDecision::Error {
                message:
                    "expected `needs-crate-type` to have a comma-separated list of crate types"
                        .to_string(),
            };
        };

        // Expect directive value to be a list of comma-separated crate-types.
        let specified_crate_types = rest
            .split(',')
            .map(|crate_type| crate_type.trim())
            .map(ToString::to_string)
            .collect::<Vec<String>>();

        for crate_type in &specified_crate_types {
            if !KNOWN_CRATE_TYPES.contains(&crate_type.as_str()) {
                return IgnoreDecision::Error {
                    message: format!(
                        "unknown crate type specified in `needs-crate-type`: `{crate_type}` is not \
                        a known crate type, known values are `{:?}`",
                        KNOWN_CRATE_TYPES
                    ),
                };
            }
        }

        let satisfies_all_crate_types = specified_crate_types
            .iter()
            .all(|specified| config.supported_crate_types().contains(specified));
        if satisfies_all_crate_types {
            return IgnoreDecision::Continue;
        } else {
            return IgnoreDecision::Ignore {
                reason: format!(
                    "skipping test as target does not support all of the crate types `{:?}`",
                    specified_crate_types
                ),
            };
        }
    }

    if !name.starts_with("needs-") {
        return IgnoreDecision::Continue;
    }

    // Handled elsewhere.
    if name == "needs-llvm-components" {
        return IgnoreDecision::Continue;
    }

    let mut found_valid = false;
    for need in needs {
        if need.name == name {
            if need.condition {
                found_valid = true;
                break;
            } else {
                return IgnoreDecision::Ignore {
                    reason: if let Some(comment) = rest {
                        format!("{} ({})", need.ignore_reason, comment.trim())
                    } else {
                        need.ignore_reason.into()
                    },
                };
            }
        }
    }

    if found_valid {
        IgnoreDecision::Continue
    } else {
        IgnoreDecision::Error { message: format!("invalid needs directive: {name}") }
    }
}

struct Need {
    name: &'static str,
    condition: bool,
    ignore_reason: &'static str,
}

pub(super) struct CachedNeedsConditions {
    sanitizer_support: bool,
    sanitizer_address: bool,
    sanitizer_cfi: bool,
    sanitizer_dataflow: bool,
    sanitizer_kcfi: bool,
    sanitizer_kasan: bool,
    sanitizer_leak: bool,
    sanitizer_memory: bool,
    sanitizer_thread: bool,
    sanitizer_hwaddress: bool,
    sanitizer_memtag: bool,
    sanitizer_shadow_call_stack: bool,
    sanitizer_safestack: bool,
    xray: bool,
    rust_lld: bool,
    dlltool: bool,
    symlinks: bool,
    /// Whether LLVM built with zstd, for the `needs-llvm-zstd` directive.
    llvm_zstd: bool,
}

impl CachedNeedsConditions {
    pub(super) fn load(config: &Config) -> Self {
        let target = &&*config.target;
        let sanitizers = &config.target_cfg().sanitizers;
        Self {
            sanitizer_support: std::env::var_os("RUSTC_SANITIZER_SUPPORT").is_some(),
            sanitizer_address: sanitizers.contains(&Sanitizer::Address),
            sanitizer_cfi: sanitizers.contains(&Sanitizer::Cfi),
            sanitizer_dataflow: sanitizers.contains(&Sanitizer::Dataflow),
            sanitizer_kcfi: sanitizers.contains(&Sanitizer::Kcfi),
            sanitizer_kasan: sanitizers.contains(&Sanitizer::KernelAddress),
            sanitizer_leak: sanitizers.contains(&Sanitizer::Leak),
            sanitizer_memory: sanitizers.contains(&Sanitizer::Memory),
            sanitizer_thread: sanitizers.contains(&Sanitizer::Thread),
            sanitizer_hwaddress: sanitizers.contains(&Sanitizer::Hwaddress),
            sanitizer_memtag: sanitizers.contains(&Sanitizer::Memtag),
            sanitizer_shadow_call_stack: sanitizers.contains(&Sanitizer::ShadowCallStack),
            sanitizer_safestack: sanitizers.contains(&Sanitizer::Safestack),
            xray: config.target_cfg().xray,

            // For tests using the `needs-rust-lld` directive (e.g. for `-Clink-self-contained=+linker`),
            // we need to find whether `rust-lld` is present in the compiler under test.
            //
            // The --compile-lib-path is the path to host shared libraries, but depends on the OS. For
            // example:
            // - on linux, it can be <sysroot>/lib
            // - on windows, it can be <sysroot>/bin
            //
            // However, `rust-lld` is only located under the lib path, so we look for it there.
            rust_lld: config
                .compile_lib_path
                .parent()
                .expect("couldn't traverse to the parent of the specified --compile-lib-path")
                .join("lib")
                .join("rustlib")
                .join(target)
                .join("bin")
                .join(if config.host.contains("windows") { "rust-lld.exe" } else { "rust-lld" })
                .exists(),

            llvm_zstd: llvm_has_libzstd(&config),
            dlltool: find_dlltool(&config),
            symlinks: has_symlinks(),
        }
    }
}

fn find_dlltool(config: &Config) -> bool {
    let path = std::env::var_os("PATH").expect("missing PATH environment variable");
    let path = std::env::split_paths(&path).collect::<Vec<_>>();

    // dlltool is used ony by GNU based `*-*-windows-gnu`
    if !(config.matches_os("windows") && config.matches_env("gnu") && config.matches_abi("")) {
        return false;
    }

    // On Windows, dlltool.exe is used for all architectures.
    // For non-Windows, there are architecture specific dlltool binaries.
    let dlltool_found = if cfg!(windows) {
        path.iter().any(|dir| dir.join("dlltool.exe").is_file())
    } else if config.matches_arch("i686") {
        path.iter().any(|dir| dir.join("i686-w64-mingw32-dlltool").is_file())
    } else if config.matches_arch("x86_64") {
        path.iter().any(|dir| dir.join("x86_64-w64-mingw32-dlltool").is_file())
    } else {
        false
    };
    dlltool_found
}

// FIXME(#135928): this is actually not quite right because this detection is run on the **host**.
// This however still helps the case of windows -> windows local development in case symlinks are
// not available.
#[cfg(windows)]
fn has_symlinks() -> bool {
    if std::env::var_os("CI").is_some() {
        return true;
    }
    let link = std::env::temp_dir().join("RUST_COMPILETEST_SYMLINK_CHECK");
    if std::os::windows::fs::symlink_file("DOES NOT EXIST", &link).is_ok() {
        std::fs::remove_file(&link).unwrap();
        true
    } else {
        false
    }
}

#[cfg(not(windows))]
fn has_symlinks() -> bool {
    true
}
