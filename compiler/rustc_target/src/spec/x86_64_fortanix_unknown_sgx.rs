use std::iter;

use super::{LinkerFlavor, LldFlavor, PanicStrategy, Target, TargetOptions};

pub fn target() -> Target {
    const PRE_LINK_ARGS: &[&str] = &[
        "--as-needed",
        "-z",
        "noexecstack",
        "-e",
        "elf_entry",
        "-Bstatic",
        "--gc-sections",
        "-z",
        "text",
        "-z",
        "norelro",
        "--no-undefined",
        "--error-unresolved-symbols",
        "--no-undefined-version",
        "-Bsymbolic",
        "--export-dynamic",
        // The following symbols are needed by libunwind, which is linked after
        // libstd. Make sure they're included in the link.
        "-u",
        "__rust_abort",
        "-u",
        "__rust_c_alloc",
        "-u",
        "__rust_c_dealloc",
        "-u",
        "__rust_print_err",
        "-u",
        "__rust_rwlock_rdlock",
        "-u",
        "__rust_rwlock_unlock",
        "-u",
        "__rust_rwlock_wrlock",
    ];

    const EXPORT_SYMBOLS: &[&str] = &[
        "sgx_entry",
        "HEAP_BASE",
        "HEAP_SIZE",
        "RELA",
        "RELACOUNT",
        "ENCLAVE_SIZE",
        "CFGDATA_BASE",
        "DEBUG",
        "EH_FRM_HDR_OFFSET",
        "EH_FRM_HDR_LEN",
        "EH_FRM_OFFSET",
        "EH_FRM_LEN",
        "TEXT_BASE",
        "TEXT_SIZE",
    ];
    let opts = TargetOptions {
        target_os: "unknown".into(),
        target_env: "sgx".into(),
        target_vendor: "fortanix".into(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),
        dynamic_linking: false,
        executables: true,
        linker_is_gnu: true,
        linker: Some("rust-lld".to_owned()),
        max_atomic_width: Some(64),
        panic_strategy: PanicStrategy::Unwind,
        cpu: "x86-64".into(),
        features: "+rdrnd,+rdseed,+lvi-cfi,+lvi-load-hardening".into(),
        llvm_args: vec!["--x86-experimental-lvi-inline-asm-hardening".into()],
        position_independent_executables: true,
        pre_link_args: iter::once((
            LinkerFlavor::Lld(LldFlavor::Ld),
            PRE_LINK_ARGS.iter().cloned().map(String::from).collect(),
        ))
        .collect(),
        override_export_symbols: Some(EXPORT_SYMBOLS.iter().cloned().map(String::from).collect()),
        relax_elf_relocations: true,
        ..Default::default()
    };
    Target {
        llvm_target: "x86_64-elf".into(),
        pointer_width: 64,
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
            .into(),
        arch: "x86_64".into(),
        options: opts,
    }
}
