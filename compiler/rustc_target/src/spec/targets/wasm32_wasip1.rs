//! The `wasm32-wasip1` enables compiling to WebAssembly using the first
//! version of the WASI standard, called "preview1". This version of the
//! standard was never formally specified and WASI has since evolved to a
//! "preview2". This target in rustc uses the previous version of the proposal.
//!
//! This target uses the syscalls defined at
//! <https://github.com/WebAssembly/WASI/tree/main/legacy/preview1>.
//!
//! Note that this target was historically called `wasm32-wasi` originally and
//! was since renamed to `wasm32-wasip1` after the preview2 target was
//! introduced.

use crate::spec::{
    Cc, LinkSelfContainedDefault, LinkerFlavor, Target, TargetMetadata, base, crt_objects,
};

pub(crate) fn target() -> Target {
    let mut options = base::wasm::options();

    options.os = "wasi".into();
    options.env = "p1".into();
    options.add_pre_link_args(LinkerFlavor::WasmLld(Cc::Yes), &["--target=wasm32-wasip1"]);

    options.pre_link_objects_self_contained = crt_objects::pre_wasi_self_contained();
    options.post_link_objects_self_contained = crt_objects::post_wasi_self_contained();

    // FIXME: Figure out cases in which WASM needs to link with a native toolchain.
    options.link_self_contained = LinkSelfContainedDefault::True;

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

    // WASI's `sys::args::init` function ignores its arguments; instead,
    // `args::args()` makes the WASI API calls itself.
    options.main_needs_argc_argv = false;

    // And, WASI mangles the name of "main" to distinguish between different
    // signatures.
    options.entry_name = "__main_void".into();

    Target {
        llvm_target: "wasm32-wasip1".into(),
        metadata: TargetMetadata {
            description: Some("WebAssembly with WASI".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-i128:128-n32:64-S128-ni:1:10:20".into(),
        arch: "wasm32".into(),
        options,
    }
}
