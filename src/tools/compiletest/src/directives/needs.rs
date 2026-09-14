use std::collections::HashMap;

use crate::common::{
    Config, KNOWN_CRATE_TYPES, KNOWN_TARGET_HAS_ATOMIC_WIDTHS, Sanitizer, query_rustc_output,
};
use crate::directives::{DirectiveLine, IgnoreDecision, KNOWN_DIRECTIVE_NAMES_SET};

pub(super) fn handle_needs(
    conditions: &PreparedNeedsConditions,
    config: &Config,
    ln: &DirectiveLine<'_>,
) -> IgnoreDecision {
    let &DirectiveLine { name, .. } = ln;

    if !name.starts_with("needs-") {
        return IgnoreDecision::Continue;
    }

    if name == "needs-target-has-atomic" {
        let Some(rest) = ln.value_after_colon() else {
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
        let Some(rest) = ln.value_after_colon() else {
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

    // Handled elsewhere.
    if name == "needs-llvm-components" || name == "needs-backends" {
        return IgnoreDecision::Continue;
    }

    if let Some(need) = conditions.simple_needs.get(name) {
        if need.condition {
            IgnoreDecision::Continue
        } else {
            IgnoreDecision::Ignore {
                reason: if let Some(comment) = ln.remark_after_space() {
                    format!("{} ({})", need.ignore_reason, comment.trim())
                } else {
                    need.ignore_reason.into()
                },
            }
        }
    } else {
        IgnoreDecision::Error { message: format!("invalid needs directive: {name}") }
    }
}

struct Need {
    name: &'static str,
    condition: bool,
    ignore_reason: &'static str,
}

pub(crate) struct PreparedNeedsConditions {
    /// The `//@ needs-*` conditions that can be treated as a simple name->boolean mapping.
    simple_needs: HashMap<&'static str, Need>,
}

pub(crate) fn prepare_needs_conditions(config: &Config) -> PreparedNeedsConditions {
    let target = config.target.as_str();
    let sanitizers = config.target_cfg().sanitizers.as_slice();

    // Note that we intentionally still put the needs- prefix here to make the file show up when
    // grepping for a directive name, even though we could technically strip that.
    let simple_needs = vec![
        // This used to be a more general `//@ needs-asm-mnemonic: ret` directive,
        // but was simplified to just `//@ needs-asm-ret` because there are very
        // few other mnemonics (`nop`?) that it could ever be useful with.
        Need {
            name: "needs-asm-ret",
            condition: has_mnemonic(config, "ret"),
            ignore_reason: "ignored on targets without a `ret` assembly instruction",
        },
        Need {
            name: "needs-asm-support",
            condition: config.has_asm_support(),
            ignore_reason: "ignored on targets without inline assembly support",
        },
        Need {
            name: "needs-sanitizer-support",
            condition: std::env::var_os("RUSTC_SANITIZER_SUPPORT").is_some(),
            ignore_reason: "ignored on targets without sanitizers support",
        },
        Need {
            name: "needs-sanitizer-address",
            condition: sanitizers.contains(&Sanitizer::Address),
            ignore_reason: "ignored on targets without address sanitizer",
        },
        Need {
            name: "needs-sanitizer-cfi",
            condition: sanitizers.contains(&Sanitizer::Cfi),
            ignore_reason: "ignored on targets without CFI sanitizer",
        },
        Need {
            name: "needs-sanitizer-dataflow",
            condition: sanitizers.contains(&Sanitizer::Dataflow),
            ignore_reason: "ignored on targets without dataflow sanitizer",
        },
        Need {
            name: "needs-sanitizer-kcfi",
            condition: sanitizers.contains(&Sanitizer::Kcfi),
            ignore_reason: "ignored on targets without kernel CFI sanitizer",
        },
        Need {
            name: "needs-sanitizer-kasan",
            condition: sanitizers.contains(&Sanitizer::KernelAddress),
            ignore_reason: "ignored on targets without kernel address sanitizer",
        },
        Need {
            name: "needs-sanitizer-khwasan",
            condition: sanitizers.contains(&Sanitizer::KernelHwaddress),
            ignore_reason: "ignored on targets without kernel hardware-assisted address sanitizer",
        },
        Need {
            name: "needs-sanitizer-leak",
            condition: sanitizers.contains(&Sanitizer::Leak),
            ignore_reason: "ignored on targets without leak sanitizer",
        },
        Need {
            name: "needs-sanitizer-memory",
            condition: sanitizers.contains(&Sanitizer::Memory),
            ignore_reason: "ignored on targets without memory sanitizer",
        },
        Need {
            name: "needs-sanitizer-thread",
            condition: sanitizers.contains(&Sanitizer::Thread),
            ignore_reason: "ignored on targets without thread sanitizer",
        },
        Need {
            name: "needs-sanitizer-hwaddress",
            condition: sanitizers.contains(&Sanitizer::Hwaddress),
            ignore_reason: "ignored on targets without hardware-assisted address sanitizer",
        },
        Need {
            name: "needs-sanitizer-memtag",
            condition: sanitizers.contains(&Sanitizer::Memtag),
            ignore_reason: "ignored on targets without memory tagging sanitizer",
        },
        Need {
            name: "needs-sanitizer-realtime",
            condition: sanitizers.contains(&Sanitizer::Realtime),
            ignore_reason: "ignored on targets without realtime sanitizer",
        },
        Need {
            name: "needs-sanitizer-shadow-call-stack",
            condition: sanitizers.contains(&Sanitizer::ShadowCallStack),
            ignore_reason: "ignored on targets without shadow call stacks",
        },
        Need {
            name: "needs-sanitizer-safestack",
            condition: sanitizers.contains(&Sanitizer::Safestack),
            ignore_reason: "ignored on targets without SafeStack support",
        },
        Need {
            name: "needs-enzyme",
            condition: config.has_enzyme && config.default_codegen_backend.is_llvm(),
            ignore_reason: "ignored when LLVM Enzyme is disabled or LLVM is not the default codegen backend",
        },
        Need {
            name: "needs-offload",
            condition: config.has_offload && config.default_codegen_backend.is_llvm(),
            ignore_reason: "ignored when LLVM Offload is disabled or LLVM is not the default codegen backend",
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
            condition: config.target_cfg().xray,
            ignore_reason: "ignored on targets without xray tracing",
        },
        Need {
            name: "needs-rust-lld",
            condition: {
                // For tests using the `needs-rust-lld` directive (e.g. for `-Clink-self-contained=+linker`),
                // we need to find whether `rust-lld` is present in the compiler under test.
                //
                // The --compile-lib-path is the path to host shared libraries, but depends on the OS. For
                // example:
                // - on linux, it can be <sysroot>/lib
                // - on windows, it can be <sysroot>/bin
                //
                // However, `rust-lld` is only located under the lib path, so we look for it there.
                config
                    .host_compile_lib_path
                    .parent()
                    .expect("couldn't traverse to the parent of the specified --compile-lib-path")
                    .join("lib")
                    .join("rustlib")
                    .join(target)
                    .join("bin")
                    .join(if config.host.contains("windows") { "rust-lld.exe" } else { "rust-lld" })
                    .exists()
            },
            ignore_reason: "ignored on targets without Rust's LLD",
        },
        Need {
            name: "needs-dlltool",
            condition: find_dlltool(config),
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
            condition: has_symlinks(),
            ignore_reason: "ignored if symlinks are unavailable",
        },
        Need {
            name: "needs-llvm-zstd",
            condition: config.default_codegen_backend.is_llvm() && llvm_has_zstd(config),
            ignore_reason: "ignored if LLVM wasn't build with zstd for ELF section compression or LLVM is not the default codegen backend",
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
        Need {
            name: "needs-std-remap-debuginfo",
            condition: config.with_std_remap_debuginfo,
            ignore_reason: "ignored if std wasn't built with remapping of debuginfo",
        },
        Need {
            name: "needs-target-std",
            condition: build_helper::targets::target_supports_std(&config.target),
            ignore_reason: "ignored if target does not support std",
        },
    ];
    let simple_needs = simple_needs
        .into_iter()
        .map(|need| {
            let name = need.name;
            assert!(name.starts_with("needs-"), "must start with `needs-`: {name:?}");
            assert!(KNOWN_DIRECTIVE_NAMES_SET.contains(name), "unknown directive name: {name:?}");
            (name, need)
        })
        .collect::<HashMap<_, _>>();

    PreparedNeedsConditions { simple_needs }
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

fn llvm_has_zstd(config: &Config) -> bool {
    // FIXME(#149764): This actually queries the compiler's _default_ backend,
    // which is usually LLVM, but can be another backend depending on the value
    // of `rust.codegen-backends` in bootstrap.toml.

    // The compiler already knows whether LLVM was built with zstd or not,
    // so compiletest can just ask the compiler.
    let output = query_rustc_output(
        config,
        &["-Zunstable-options", "--print=backend-has-zstd"],
        Default::default(),
    );
    match output.trim() {
        "true" => true,
        "false" => false,
        _ => panic!("unexpected output from `--print=backend-has-zstd`: {output:?}"),
    }
}

fn has_mnemonic(config: &Config, mnemonic: &str) -> bool {
    if !config.default_codegen_backend.is_llvm() {
        return false;
    }

    let target_flag = format!("--target={}", config.target);
    let output = query_rustc_output(
        config,
        &[
            &target_flag,
            "-Zunstable-options",
            &format!("--print=backend-has-mnemonic:{}", mnemonic),
        ],
        Default::default(),
    );

    match output.trim() {
        "true" => true,
        "false" => false,
        _ => panic!("unexpected output from `--print=backend-has-mnemonic:{mnemonic}`: {output:?}"),
    }
}
