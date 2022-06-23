use super::{cvs, wasm_base};
use super::{LinkArgs, LinkerFlavor, PanicStrategy, Target, TargetOptions};

pub fn target() -> Target {
    let mut options = wasm_base::options();

    let clang_args = options.pre_link_args.entry(LinkerFlavor::Gcc).or_default();

    // Rust really needs a way for users to specify exports and imports in
    // the source code. --export-dynamic isn't the right tool for this job,
    // however it does have the side effect of automatically exporting a lot
    // of symbols, which approximates what people want when compiling for
    // wasm32-unknown-unknown expect, so use it for now.
    clang_args.push("--export-dynamic".into());

    let mut post_link_args = LinkArgs::new();
    post_link_args.insert(
        LinkerFlavor::Em,
        vec!["-sABORTING_MALLOC=0".into(), "-Wl,--fatal-warnings".into()],
    );

    let opts = TargetOptions {
        os: "emscripten".into(),
        linker_flavor: LinkerFlavor::Em,
        // emcc emits two files - a .js file to instantiate the wasm and supply platform
        // functionality, and a .wasm file.
        exe_suffix: ".js".into(),
        linker: None,
        is_like_emscripten: true,
        panic_strategy: PanicStrategy::Unwind,
        no_default_libraries: false,
        post_link_args,
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
