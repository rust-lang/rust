use super::wasm32_base;
use super::{LinkArgs, LinkerFlavor, PanicStrategy, Target, TargetOptions};

pub fn target() -> Target {
    let mut post_link_args = LinkArgs::new();
    post_link_args.insert(
        LinkerFlavor::Em,
        vec![
            "-s".to_string(),
            "ERROR_ON_UNDEFINED_SYMBOLS=1".to_string(),
            "-s".to_string(),
            "ASSERTIONS=1".to_string(),
            "-s".to_string(),
            "ABORTING_MALLOC=0".to_string(),
            "-Wl,--fatal-warnings".to_string(),
        ],
    );

    let opts = TargetOptions {
        os: "emscripten".to_string(),
        linker_flavor: LinkerFlavor::Em,
        // emcc emits two files - a .js file to instantiate the wasm and supply platform
        // functionality, and a .wasm file.
        exe_suffix: ".js".to_string(),
        linker: None,
        linker_is_gnu: true,
        is_like_emscripten: true,
        panic_strategy: PanicStrategy::Unwind,
        post_link_args,
        os_family: Some("unix".to_string()),
        ..wasm32_base::options()
    };
    Target {
        llvm_target: "wasm32-unknown-emscripten".to_string(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-i64:64-n32:64-S128".to_string(),
        arch: "wasm32".to_string(),
        options: opts,
    }
}
