//! A "bare wasm" target representing a WebAssembly output that does not import
//! anything from its environment and also specifies an _upper_ bound on the set
//! of WebAssembly proposals that are supported.
//!
//! It's equivalent to the `wasm32-unknown-unknown` target with the additional
//! flags `-Ctarget-cpu=mvp` and `-Ctarget-feature=+mutable-globals`. This
//! enables just the features specified in <https://www.w3.org/TR/wasm-core-1/>
//!
//! This is a _separate target_ because using `wasm32-unknown-unknown` with
//! those target flags doesn't automatically rebuild libcore / liballoc with
//! them, and in order to get those libraries rebuilt you need to use the
//! nightly Rust feature `-Zbuild-std`. This target is for people who want to
//! use stable Rust, and target a stable set pf WebAssembly features.

use crate::spec::{Cc, LinkerFlavor, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut options = base::wasm::options();
    options.os = "none".into();

    // WebAssembly 1.0 shipped in 2019 and included exactly one proposal
    // after the initial "MVP" feature set: "mutable-globals".
    options.cpu = "mvp".into();
    options.features = "+mutable-globals".into();

    options.add_pre_link_args(
        LinkerFlavor::WasmLld(Cc::No),
        &[
            // For now this target just never has an entry symbol no matter the output
            // type, so unconditionally pass this.
            "--no-entry",
        ],
    );
    options.add_pre_link_args(
        LinkerFlavor::WasmLld(Cc::Yes),
        &[
            // Make sure clang uses LLD as its linker and is configured appropriately
            // otherwise
            "--target=wasm32-unknown-unknown",
            "-Wl,--no-entry",
        ],
    );

    Target {
        llvm_target: "wasm32-unknown-unknown".into(),
        metadata: TargetMetadata {
            description: Some("WebAssembly".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-i128:128-n32:64-S128-ni:1:10:20".into(),
        arch: "wasm32".into(),
        options,
    }
}
