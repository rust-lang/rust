use super::{cvs, wasm_base};
use super::{LinkerFlavor, PanicStrategy, RelocModel, Target, TargetOptions};

pub fn target() -> Target {
    let mut options = wasm_base::options();

    // Rust really needs a way for users to specify exports and imports in
    // the source code. --export-dynamic isn't the right tool for this job,
    // however it does have the side effect of automatically exporting a lot
    // of symbols, which approximates what people want when compiling for
    // wasm32-unknown-unknown expect, so use it for now.
    options.add_pre_link_args(LinkerFlavor::Gcc, &["--export-dynamic"]);
    options.add_post_link_args(LinkerFlavor::Em, &["-sABORTING_MALLOC=0", "-Wl,--fatal-warnings"]);

    let opts = TargetOptions {
        os: "emscripten".into(),
        linker_flavor: LinkerFlavor::Em,
        // emcc emits two files - a .js file to instantiate the wasm and supply platform
        // functionality, and a .wasm file.
        exe_suffix: ".js".into(),
        linker: None,
        relocation_model: RelocModel::Pic,
        panic_strategy: PanicStrategy::Unwind,
        no_default_libraries: false,
        families: cvs!["unix", "wasm"],
        ..options
    };
    Target {
        llvm_target: "wasm32-unknown-emscripten".into(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-f128:64-n32:64-S128-ni:1:10:20".into(),
        arch: "wasm32".into(),
        options: opts,
    }
}
