use crate::spec::{PanicStrategy, Target, TargetMetadata, TargetOptions};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "hexagon-unknown-none-elf".into(),
        metadata: TargetMetadata {
            description: Some("Bare Hexagon (v60+, HVX)".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
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
            cpu: "hexagonv60".into(),
            panic_strategy: PanicStrategy::Abort,
            dynamic_linking: true,
            features: "-small-data,+hvx-length128b".into(),
            max_atomic_width: Some(32),
            emit_debug_gdb_scripts: false,
            c_enum_min_bits: Some(8),
            ..Default::default()
        },
    }
}
