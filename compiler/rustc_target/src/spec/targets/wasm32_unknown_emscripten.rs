use crate::spec::{Arch, StdCrate, StdSupport, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "wasm32-unknown-emscripten".into(),
        metadata: TargetMetadata {
            description: Some("WebAssembly via Emscripten".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
            standard_library_support: Some(StdSupport::from_crate(StdCrate::Std)),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-i128:128-f128:64-n32:64-S128-ni:1:10:20"
            .into(),
        arch: Arch::Wasm32,
        options: base::wasm::emscripten_options(),
    }
}
