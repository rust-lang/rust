use std::iter;

use super::{LinkerFlavor, PanicStrategy, Target, TargetOptions};

pub fn target() -> Result<Target, String> {
    const PRE_LINK_ARGS: &[&str] = &[
        "-Wl,--as-needed",
        "-Wl,-z,noexecstack",
        "-m64",
        "-fuse-ld=gold",
        "-nostdlib",
        "-shared",
        "-Wl,-e,sgx_entry",
        "-Wl,-Bstatic",
        "-Wl,--gc-sections",
        "-Wl,-z,text",
        "-Wl,-z,norelro",
        "-Wl,--rosegment",
        "-Wl,--no-undefined",
        "-Wl,--error-unresolved-symbols",
        "-Wl,--no-undefined-version",
        "-Wl,-Bsymbolic",
        "-Wl,--export-dynamic",
        // The following symbols are needed by libunwind, which is linked after
        // libstd. Make sure they're included in the link.
        "-Wl,-u,__rust_abort",
        "-Wl,-u,__rust_c_alloc",
        "-Wl,-u,__rust_c_dealloc",
        "-Wl,-u,__rust_print_err",
        "-Wl,-u,__rust_rwlock_rdlock",
        "-Wl,-u,__rust_rwlock_unlock",
        "-Wl,-u,__rust_rwlock_wrlock",
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
        "EH_FRM_HDR_BASE",
        "EH_FRM_HDR_SIZE",
        "TEXT_BASE",
        "TEXT_SIZE",
    ];
    let opts = TargetOptions {
        dynamic_linking: false,
        executables: true,
        linker_is_gnu: true,
        max_atomic_width: Some(64),
        panic_strategy: PanicStrategy::Unwind,
        cpu: "x86-64".into(),
        features: "+rdrnd,+rdseed".into(),
        position_independent_executables: true,
        pre_link_args: iter::once((
            LinkerFlavor::Gcc,
            PRE_LINK_ARGS.iter().cloned().map(String::from).collect(),
        ))
        .collect(),
        post_link_objects: vec!["libunwind.a".into()],
        override_export_symbols: Some(EXPORT_SYMBOLS.iter().cloned().map(String::from).collect()),
        ..Default::default()
    };
    Ok(Target {
        llvm_target: "x86_64-unknown-linux-gnu".into(),
        target_endian: "little".into(),
        target_pointer_width: "64".into(),
        target_c_int_width: "32".into(),
        target_os: "unknown".into(),
        target_env: "sgx".into(),
        target_vendor: "fortanix".into(),
        data_layout: "e-m:e-i64:64-f80:128-n8:16:32:64-S128".into(),
        arch: "x86_64".into(),
        linker_flavor: LinkerFlavor::Gcc,
        options: opts,
    })
}
