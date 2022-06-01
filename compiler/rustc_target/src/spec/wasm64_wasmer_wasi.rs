//! 64-bit WebAssembly target with a full operating system based on WASI
//!

use super::wasm_base;
use super::{crt_objects, LinkerFlavor, LldFlavor, Target};

pub fn target() -> Target {
    let mut options = wasm_base::options();
    options.os = "wasi".into();
    options.vendor = "wasmer".into();
    options.linker_flavor = LinkerFlavor::Lld(LldFlavor::Wasm);
    let clang_args = options
        .pre_link_args
        .entry(LinkerFlavor::Gcc)
        .or_insert(Vec::new());

    // Make sure clang uses LLD as its linker and is configured appropriately
    // otherwise
    clang_args.push("--target=wasm64-wasi".into());
    

    options.pre_link_objects_fallback = crt_objects::pre_wasi_fallback();
    options.post_link_objects_fallback = crt_objects::post_wasi_fallback();

    // Right now this is a bit of a workaround but we're currently saying that
    // the target by default has a static crt which we're taking as a signal
    // for "use the bundled crt". If that's turned off then the system's crt
    // will be used, but this means that default usage of this target doesn't
    // need an external compiler but it's still interoperable with an external
    // compiler if configured correctly.
    options.crt_static_default = true;
    options.crt_static_respected = true;

    // Allow `+crt-static` to create a "cdylib" output which is just a wasm file
    // without a main function.
    options.crt_static_allows_dylibs = true;

    // WASIX's `sys::args::init` function ignores its arguments; instead,
    // `args::args()` makes the WASIX API calls itself.
    options.main_needs_argc_argv = false;

    let lld_args = options.pre_link_args.get_mut(&LinkerFlavor::Lld(LldFlavor::Wasm)).unwrap();
    lld_args.push("-mwasm64".into());

    // TODO: Adding "+atomics" here seems to enable more of the multithreading however it does not yet
    //       work properly so more work is needed to finish this, otherwise this is very close to
    //       full networking support.
    // We need shared memory for multithreading
    //lld_args.push("--shared-memory".into());

    // Any engine that implements wasm64 will surely implement the rest of these
    // features since they were all merged into the official spec by the time
    // wasm64 was designed.
    options.features = "+bulk-memory,+mutable-globals,+sign-ext,+nontrapping-fptoint".into();

    Target {
        llvm_target: "wasm64-wasi".into(),
        pointer_width: 64,
        data_layout: "e-m:e-p:64:64-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20".into(),
        arch: "wasm64".into(),
        options,
    }
}
