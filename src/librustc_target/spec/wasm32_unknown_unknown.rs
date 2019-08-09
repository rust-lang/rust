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
//! Group nowadays at https://github.com/rustwasm.

use super::{LldFlavor, LinkerFlavor, Target};
use super::wasm32_base;

pub fn target() -> Result<Target, String> {
    let mut options = wasm32_base::options();
    let clang_args = options.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap();

    // Make sure clang uses LLD as its linker and is configured appropriately
    // otherwise
    clang_args.push("--target=wasm32-unknown-unknown".to_string());

    // Disable attempting to link crt1.o since it typically isn't present and
    // isn't needed currently.
    clang_args.push("-nostdlib".to_string());

    // For now this target just never has an entry symbol no matter the output
    // type, so unconditionally pass this.
    clang_args.push("-Wl,--no-entry".to_string());
    options.pre_link_args.get_mut(&LinkerFlavor::Lld(LldFlavor::Wasm))
        .unwrap()
        .push("--no-entry".to_string());

    Ok(Target {
        llvm_target: "wasm32-unknown-unknown".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        target_os: "unknown".to_string(),
        target_env: String::new(),
        target_vendor: "unknown".to_string(),
        data_layout: "e-m:e-p:32:32-i64:64-n32:64-S128".to_string(),
        arch: "wasm32".to_string(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Wasm),
        options,
    })
}
