use crate::spec::{Arch, Cc, LinkerFlavor, Os, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut options = base::wasm::options();

    options.os = Os::None;
    options.linker = Some("wasm-component-ld".into());

    options.add_pre_link_args(
        LinkerFlavor::WasmLld(Cc::No),
        &[
            // For now this target just never has an entry symbol no matter the output
            // type, so unconditionally pass this.
            "--no-entry",
        ],
    );

    Target {
        llvm_target: "wasm32-none-component".into(),
        metadata: TargetMetadata {
            description: Some("WebAssembly".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-i128:128-n32:64-S128-ni:1:10:20".into(),
        arch: Arch::Wasm32,
        options,
    }
}
