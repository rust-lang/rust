use super::{LinkArgs, LinkerFlavor, Target, TargetOptions};

pub fn target() -> Result<Target, String> {
    let mut post_link_args = LinkArgs::new();
    post_link_args.insert(LinkerFlavor::Em,
                          vec!["-s".to_string(),
                               "WASM=1".to_string(),
                               "-s".to_string(),
                               "ASSERTIONS=1".to_string(),
                               "-s".to_string(),
                               "ERROR_ON_UNDEFINED_SYMBOLS=1".to_string(),
                               "-g3".to_string()]);

    let opts = TargetOptions {
        dynamic_linking: false,
        executables: true,
        // Today emcc emits two files - a .js file to bootstrap and
        // possibly interpret the wasm, and a .wasm file
        exe_suffix: ".js".to_string(),
        linker_is_gnu: true,
        link_env: vec![("EMCC_WASM_BACKEND".to_string(), "1".to_string())],
        allow_asm: false,
        obj_is_bitcode: true,
        is_like_emscripten: true,
        max_atomic_width: Some(32),
        post_link_args,
        limit_rdylib_exports: false,
        target_family: Some("unix".to_string()),
        .. Default::default()
    };
    Ok(Target {
        llvm_target: "wasm32-unknown-unknown".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        target_os: "emscripten".to_string(),
        target_env: String::new(),
        target_vendor: "unknown".to_string(),
        data_layout: "e-m:e-p:32:32-i64:64-n32:64-S128".to_string(),
        arch: "wasm32".to_string(),
        linker_flavor: LinkerFlavor::Em,
        options: opts,
    })
}
