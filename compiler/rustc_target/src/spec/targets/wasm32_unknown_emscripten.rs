use crate::spec::{
    Arch, LinkArgs, LinkerFlavor, Os, PanicStrategy, RelocModel, Target, TargetMetadata,
    TargetOptions, base, cvs,
};

pub(crate) fn target() -> Target {
    // Reset flags for non-Em flavors back to empty to satisfy sanity checking tests.
    let pre_link_args = LinkArgs::new();
    let post_link_args =
        TargetOptions::link_args(LinkerFlavor::EmCc, &["-sABORTING_MALLOC=0", "-sWASM_BIGINT"]);

    let opts = TargetOptions {
        os: Os::Emscripten,
        linker_flavor: LinkerFlavor::EmCc,
        // emcc emits two files - a .js file to instantiate the wasm and supply platform
        // functionality, and a .wasm file.
        exe_suffix: ".js".into(),
        linker: None,
        pre_link_args,
        post_link_args,
        relocation_model: RelocModel::Pic,
        // crt_static should always be true for an executable and always false
        // for a shared library. There is no easy way to indicate this and it
        // doesn't seem to matter much so we set crt_static_allows_dylibs to
        // true and leave crt_static as true when linking dynamic libraries.
        // wasi also sets crt_static_allows_dylibs: true so this is at least
        // aligned between wasm targets.
        crt_static_respected: true,
        crt_static_default: true,
        crt_static_allows_dylibs: true,
        panic_strategy: PanicStrategy::Unwind,
        no_default_libraries: false,
        families: cvs!["unix", "wasm"],
        ..base::wasm::options()
    };
    Target {
        llvm_target: "wasm32-unknown-emscripten".into(),
        metadata: TargetMetadata {
            description: Some("WebAssembly via Emscripten".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-i128:128-f128:64-n32:64-S128-ni:1:10:20"
            .into(),
        arch: Arch::Wasm32,
        options: opts,
    }
}
