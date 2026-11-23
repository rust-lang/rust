use crate::spec::{Arch, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "wasm64-unknown-emscripten".into(),
        metadata: TargetMetadata {
            description: Some("WebAssembly with Memory64 via Emscripten".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p:64:64-p10:8:8-p20:8:8-i64:64-i128:128-f128:64-n32:64-S128-ni:1:10:20"
            .into(),
        arch: Arch::Wasm64,
        options: TargetOptions {
            features: base::wasm::default_wasm64_features(),
            ..base::wasm::emscripten_options()
        },
    }
}
