//! A "bare wasm" target representing a WebAssembly output that makes zero
//! assumptions about its environment.
//!
//! The `wasm64-unknown-unknown` target is intended to encapsulate use cases
//! that do not rely on any imported functionality. The binaries generated are
//! entirely self-contained by default when using the standard library. Although
//! the standard library is available, most of it returns an error immediately
//! (e.g. trying to create a TCP stream or something like that).

use crate::spec::{add_link_args, base, Cc, LinkerFlavor, MaybeLazy, Target, TargetOptions};

pub fn target() -> Target {
    let mut options = base::wasm::options();
    options.os = "unknown".into();

    options.pre_link_args = MaybeLazy::lazy(|| {
        let mut pre_link_args = TargetOptions::link_args(
            LinkerFlavor::WasmLld(Cc::No),
            &[
                // For now this target just never has an entry symbol no matter the output
                // type, so unconditionally pass this.
                "--no-entry",
                "-mwasm64",
            ],
        );
        add_link_args(
            &mut pre_link_args,
            LinkerFlavor::WasmLld(Cc::Yes),
            &[
                // Make sure clang uses LLD as its linker and is configured appropriately
                // otherwise
                "--target=wasm64-unknown-unknown",
                "-Wl,--no-entry",
            ],
        );
        pre_link_args
    });

    // Any engine that implements wasm64 will surely implement the rest of these
    // features since they were all merged into the official spec by the time
    // wasm64 was designed.
    options.features = "+bulk-memory,+mutable-globals,+sign-ext,+nontrapping-fptoint".into();

    Target {
        llvm_target: "wasm64-unknown-unknown".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 64,
        data_layout: "e-m:e-p:64:64-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20".into(),
        arch: "wasm64".into(),
        options,
    }
}
