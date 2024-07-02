//! The `wasm32-wali-linux-musl` target is a wasm32 target compliant with the
//! [WebAssembly Linux Interface](https://github.com/arjunr2/WALI).

use crate::spec::{base, Cc, LinkerFlavor, Target};

pub fn target() -> Target {
    let mut options = base::linux_wasm::opts();

    options.add_pre_link_args(
        LinkerFlavor::WasmLld(Cc::No),
        &["--export-memory", "--shared-memory"],
    );
    options.add_pre_link_args(
        LinkerFlavor::WasmLld(Cc::Yes),
        &[
            "--target=wasm32-wasi-threads",
            "-Wl,--export-memory,",
            "-Wl,--shared-memory",
        ],
    );

    Target {
        llvm_target: "wasm32-wasi".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20".into(),
        arch: "wasm32".into(),
        options,
    }
}
