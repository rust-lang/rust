use crate::spec::{Cc, LinkerFlavor, Target, TargetMetadata, TargetOptions, cvs};

pub(crate) fn target() -> Target {
    let mut pre_link_args = std::collections::BTreeMap::<LinkerFlavor, Vec<_>>::new();
    pre_link_args.entry(LinkerFlavor::Unix(Cc::Yes)).or_default().extend(["-G0".into()]);

    Target {
        llvm_target: "hexagon-unknown-elf".into(),
        metadata: TargetMetadata {
            description: Some("Hexagon QuRT (Qualcomm Real-Time OS)".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: concat!(
            "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32",
            ":32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32",
            ":32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048",
            ":2048:2048"
        )
        .into(),
        arch: "hexagon".into(),
        options: TargetOptions {
            os: "qurt".into(),
            vendor: "unknown".into(),
            cpu: "hexagonv69".into(),
            linker: Some("rust-lld".into()),
            linker_flavor: LinkerFlavor::Unix(Cc::Yes),
            exe_suffix: ".elf".into(),
            dynamic_linking: true,
            executables: true,
            families: cvs!["unix"],
            has_thread_local: true,
            has_rpath: false,
            crt_static_default: false,
            crt_static_respected: true,
            crt_static_allows_dylibs: true,
            no_default_libraries: false,
            max_atomic_width: Some(32),
            features: "-small-data,+hvx-length128b".into(),
            c_enum_min_bits: Some(8),
            pre_link_args,
            ..Default::default()
        },
    }
}
