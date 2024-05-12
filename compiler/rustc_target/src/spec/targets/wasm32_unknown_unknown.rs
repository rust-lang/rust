//! A "bare wasm" target representing a WebAssembly output that makes zero
//! assumptions about its environment.
//!
//! The `wasm32-unknown-unknown` target is intended to encapsulate use cases
//! that do not rely on any imported functionality. The binaries generated are
//! entirely self-contained by default when using the standard library. Although
//! the standard library is available, most of it returns an error immediately
//! (e.g. trying to create a TCP stream or something like that).
//!
//! This target is more or less managed by the Rust and WebAssembly Working
//! Group nowadays at <https://github.com/rustwasm>.

use crate::spec::add_link_args;
use crate::spec::{base, Cc, LinkerFlavor, MaybeLazy, Target, TargetOptions};

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
            ],
        );
        add_link_args(
            &mut pre_link_args,
            LinkerFlavor::WasmLld(Cc::Yes),
            &[
                // Make sure clang uses LLD as its linker and is configured appropriately
                // otherwise
                "--target=wasm32-unknown-unknown",
                "-Wl,--no-entry",
            ],
        );
        pre_link_args
    });

    Target {
        llvm_target: "wasm32-unknown-unknown".into(),
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
