//! The `wasm32-wasix` implements a full operating system for Web Assembly
//! including multithreading and networking support.

use super::wasm_base;
use super::{crt_objects, LinkerFlavor, LldFlavor, Target};

pub fn target() -> Target {
    let mut options = wasm_base::options();

    options.os = "wasix".into();
    options.linker_flavor = LinkerFlavor::Lld(LldFlavor::Wasm);
    options
        .pre_link_args
        .entry(LinkerFlavor::Gcc)
        .or_insert(Vec::new())
        .push("--target=wasm32-wasix".into());

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

    Target {
        llvm_target: "wasm32-wasix".into(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20".into(),
        arch: "wasm32".into(),
        options,
    }
}
