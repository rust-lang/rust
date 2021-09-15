use super::{LinkerFlavor, LldFlavor, PanicStrategy, RelroLevel, Target, TargetOptions};

/// A base target for Nintendo Switch devices using a pure LLVM toolchain.
pub fn target() -> Target {
    let mut opts = TargetOptions {
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),
        linker: Some("rust-lld".into()),
        os: "horizon".into(),
        max_atomic_width: Some(128),
        panic_strategy: PanicStrategy::Abort,
        position_independent_executables: true,
        crt_static_default: false,
        crt_static_respected: false,
        dynamic_linking: true,
        executables: true,
        has_elf_tls: false,
        has_rpath: false,
        relro_level: RelroLevel::Off,
        ..Default::default()
    };

    opts.pre_link_args.insert(LinkerFlavor::Lld(LldFlavor::Ld), vec![]);

    opts.post_link_args.insert(LinkerFlavor::Lld(LldFlavor::Ld), vec![]);

    Target {
        llvm_target: "aarch64-unknown-none".into(),
        pointer_width: 64,
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".into(),
        arch: "aarch64".into(),
        options: opts,
    }
}
