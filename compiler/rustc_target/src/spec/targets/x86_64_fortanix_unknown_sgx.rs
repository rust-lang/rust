use std::borrow::Cow;

use crate::spec::{Cc, LinkerFlavor, Lld, Target, TargetMetadata, TargetOptions, cvs};

pub(crate) fn target() -> Target {
    let pre_link_args = TargetOptions::link_args(
        LinkerFlavor::Gnu(Cc::No, Lld::No),
        &[
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
        ],
    );

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
        os: "unknown".into(),
        env: "sgx".into(),
        vendor: "fortanix".into(),
        abi: "fortanix".into(),
        linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
        linker: Some("rust-lld".into()),
        max_atomic_width: Some(64),
        cpu: "x86-64".into(),
        plt_by_default: false,
        features: "+rdrnd,+rdseed,+lvi-cfi,+lvi-load-hardening".into(),
        llvm_args: cvs!["--x86-experimental-lvi-inline-asm-hardening"],
        position_independent_executables: true,
        pre_link_args,
        override_export_symbols: Some(EXPORT_SYMBOLS.iter().cloned().map(Cow::from).collect()),
        relax_elf_relocations: true,
        ..Default::default()
    };
    Target {
        llvm_target: "x86_64-elf".into(),
        metadata: TargetMetadata {
            description: Some("Fortanix ABI for 64-bit Intel SGX".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout:
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: "x86_64".into(),
        options: opts,
    }
}
