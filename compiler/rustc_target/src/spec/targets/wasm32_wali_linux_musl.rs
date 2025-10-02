//! The `wasm32-wali-linux-musl` target is a wasm32 target compliant with the
//! [WebAssembly Linux Interface](https://github.com/arjunr2/WALI).

use crate::spec::{Cc, LinkerFlavor, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut options = base::linux_wasm::opts();

    options.add_pre_link_args(
        LinkerFlavor::WasmLld(Cc::No),
        &["--export-memory", "--shared-memory", "--max-memory=1073741824"],
    );
    options.add_pre_link_args(
        LinkerFlavor::WasmLld(Cc::Yes),
        &[
            "--target=wasm32-wasi-threads",
            "-Wl,--export-memory,",
            "-Wl,--shared-memory",
            "-Wl,--max-memory=1073741824",
        ],
    );

    Target {
        llvm_target: "wasm32-wasi".into(),
        metadata: TargetMetadata {
            description: Some("WebAssembly Linux Interface with musl-libc".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: None,
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-i128:128-n32:64-S128-ni:1:10:20".into(),
        arch: "wasm32".into(),
        options,
    }
}
