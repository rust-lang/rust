//! A "bare wasm" target representing a WebAssembly output that makes zero
//! assumptions about its environment.
//!
//! The `wasm64-unknown-unknown` target is intended to encapsulate use cases
//! that do not rely on any imported functionality. The binaries generated are
//! entirely self-contained by default when using the standard library. Although
//! the standard library is available, most of it returns an error immediately
//! (e.g. trying to create a TCP stream or something like that).

use super::wasm_base;
use super::{LinkerFlavor, LldFlavor, Target};

pub fn target() -> Target {
    let mut options = wasm_base::options();
    options.os = "unknown".into();
    options.linker_flavor = LinkerFlavor::Lld(LldFlavor::Wasm);
    let clang_args = options.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap();

    // Make sure clang uses LLD as its linker and is configured appropriately
    // otherwise
    clang_args.push("--target=wasm64-unknown-unknown".into());

    // For now this target just never has an entry symbol no matter the output
    // type, so unconditionally pass this.
    clang_args.push("-Wl,--no-entry".into());

    let lld_args = options.pre_link_args.get_mut(&LinkerFlavor::Lld(LldFlavor::Wasm)).unwrap();
    lld_args.push("--no-entry".into());
    lld_args.push("-mwasm64".into());

    // Any engine that implements wasm64 will surely implement the rest of these
    // features since they were all merged into the official spec by the time
    // wasm64 was designed.
    options.features = "+bulk-memory,+mutable-globals,+sign-ext,+nontrapping-fptoint".into();

    Target {
        llvm_target: "wasm64-unknown-unknown".into(),
        pointer_width: 64,
        data_layout: "e-m:e-p:64:64-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20".into(),
        arch: "wasm64".into(),
        options,
    }
}
