use super::{cvs, wasm_base};
use super::{LinkArgs, LinkerFlavor, PanicStrategy, RelocModel, Target, TargetOptions};

pub fn target() -> Target {
    // Reset flags for non-Em flavors back to empty to satisfy sanity checking tests.
    let pre_link_args = LinkArgs::new();
    let post_link_args = TargetOptions::link_args(
        LinkerFlavor::EmCc,
        &["-sABORTING_MALLOC=0", "-Wl,--fatal-warnings"],
    );

    let opts = TargetOptions {
        os: "emscripten".into(),
        linker_flavor: LinkerFlavor::EmCc,
        // emcc emits two files - a .js file to instantiate the wasm and supply platform
        // functionality, and a .wasm file.
        exe_suffix: ".js".into(),
        linker: None,
        pre_link_args,
        post_link_args,
        relocation_model: RelocModel::Pic,
        panic_strategy: PanicStrategy::Unwind,
        no_default_libraries: false,
        families: cvs!["unix", "wasm"],
        ..wasm_base::options()
    };
    Target {
        llvm_target: "wasm32-unknown-emscripten".into(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-f128:64-n32:64-S128-ni:1:10:20".into(),
        arch: "wasm32".into(),
        options: opts,
    }
}
