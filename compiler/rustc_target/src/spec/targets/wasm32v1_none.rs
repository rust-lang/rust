//! A "bare wasm" target representing a WebAssembly output that makes zero
//! assumptions about its environment, similar to wasm32-unknown-unknown, but
//! that also specifies an _upper_ bound on the set of wasm proposals that are
//! supported.
//!
//! It is implemented as a variant on LLVM's wasm32-unknown-unknown target, with
//! the additional flags `-Ctarget-cpu=mvp` and `-Ctarget-feature=+mutable-globals`.
//!
//! This target exists to resolve a tension in Rustc's choice of WebAssembly
//! proposals to support. Since most WebAssembly users are in fact _on the web_
//! and web browsers are frequently updated with support for the latest
//! features, it is reasonable for Rustc to generate wasm code that exploits new
//! WebAssembly proposals as they gain browser support. At least by default. And
//! this is what the wasm32-unknown-unknown target does, which means that the
//! _exact_ WebAssembly features that Rustc generates will change over time.
//!
//! But a different set of users -- smaller but nonetheless worth supporting --
//! are using WebAssembly in implementations that either don't get updated very
//! often, or need to prioritize stability, implementation simplicity or
//! security over feature support. This target is for them, and it promises that
//! the wasm code it generates will not go beyond the proposals/features of the
//! W3C WebAssembly core 1.0 spec, which (as far as I can tell) is approximately
//! "the wasm MVP plus mutable globals". Mutable globals was proposed in 2018
//! and made it in.
//!
//! See https://www.w3.org/TR/wasm-core-1/
//!
//! Notably this feature-set _excludes_:
//!
//!   - sign-extension operators
//!   - non-trapping / saturating float-to-int conversions
//!   - multi-value
//!   - reference types
//!   - bulk memory operations
//!   - SIMD
//!
//! These are all listed as additions in the core 2.0 spec. Also they were all
//! proposed after 2020, and core 1.0 shipped in 2019. It also excludes even
//! later proposals such as:
//!
//!   - exception handling
//!   - tail calls
//!   - extended consts
//!   - function references
//!   - multi-memory
//!   - component model
//!   - gc
//!   - threads
//!   - relaxed SIMD
//!   - custom annotations
//!   - branch hinting
//!

use crate::spec::{Cc, LinkerFlavor, Target, base};

pub(crate) fn target() -> Target {
    let mut options = base::wasm::options();
    options.os = "none".into();

    options.cpu = "mvp".into();
    options.features = "+mutable-globals".into();

    options.add_pre_link_args(LinkerFlavor::WasmLld(Cc::No), &[
        // For now this target just never has an entry symbol no matter the output
        // type, so unconditionally pass this.
        "--no-entry",
    ]);
    options.add_pre_link_args(LinkerFlavor::WasmLld(Cc::Yes), &[
        // Make sure clang uses LLD as its linker and is configured appropriately
        // otherwise
        "--target=wasm32-unknown-unknown",
        "-Wl,--no-entry",
    ]);

    Target {
        llvm_target: "wasm32-unknown-unknown".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("WebAssembly".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20".into(),
        arch: "wasm32".into(),
        options,
    }
}
