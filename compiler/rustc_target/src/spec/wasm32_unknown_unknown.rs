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

use super::wasm_base;
use super::{LinkerFlavor, LldFlavor, Target};
use crate::spec::abi::Abi;

pub fn target() -> Target {
    let mut options = wasm_base::options();
    options.os = "unknown".to_string();
    options.linker_flavor = LinkerFlavor::Lld(LldFlavor::Wasm);

    // This is a default for backwards-compatibility with the original
    // definition of this target oh-so-long-ago. Once the "wasm" ABI is
    // stable and the wasm-bindgen project has switched to using it then there's
    // no need for this and it can be removed.
    //
    // Currently this is the reason that this target's ABI is mismatched with
    // clang's ABI. This means that, in the limit, you can't merge C and Rust
    // code on this target due to this ABI mismatch.
    options.default_adjusted_cabi = Some(Abi::Wasm);

    let clang_args = options.pre_link_args.entry(LinkerFlavor::Gcc).or_default();

    // Make sure clang uses LLD as its linker and is configured appropriately
    // otherwise
    clang_args.push("--target=wasm32-unknown-unknown".to_string());

    // For now this target just never has an entry symbol no matter the output
    // type, so unconditionally pass this.
    clang_args.push("-Wl,--no-entry".to_string());

    // Rust really needs a way for users to specify exports and imports in
    // the source code. --export-dynamic isn't the right tool for this job,
    // however it does have the side effect of automatically exporting a lot
    // of symbols, which approximates what people want when compiling for
    // wasm32-unknown-unknown expect, so use it for now.
    clang_args.push("-Wl,--export-dynamic".to_string());

    // Add the flags to wasm-ld's args too.
    let lld_args = options.pre_link_args.entry(LinkerFlavor::Lld(LldFlavor::Wasm)).or_default();
    lld_args.push("--no-entry".to_string());
    lld_args.push("--export-dynamic".to_string());

    Target {
        llvm_target: "wasm32-unknown-unknown".to_string(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20".to_string(),
        arch: "wasm32".to_string(),
        options,
    }
}
