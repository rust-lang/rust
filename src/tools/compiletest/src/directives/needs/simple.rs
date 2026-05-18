use crate::common::Config;
use crate::directives::CachedNeedsConditions;
use crate::directives::needs::Need;

pub(crate) fn simple_needs(cache: &CachedNeedsConditions, config: &Config) -> Vec<Need> {
    // Note that we intentionally still put the needs- prefix here to make the file show up when
    // grepping for a directive name, even though we could technically strip that.
    vec![
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
            name: "needs-sanitizer-khwasan",
            condition: cache.sanitizer_khwasan,
            ignore_reason: "ignored on targets without kernel hardware-assisted address sanitizer",
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
            name: "needs-sanitizer-realtime",
            condition: cache.sanitizer_realtime,
            ignore_reason: "ignored on targets without realtime sanitizer",
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
            condition: cache.llvm_zstd && config.default_codegen_backend.is_llvm(),
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
    ]
}
