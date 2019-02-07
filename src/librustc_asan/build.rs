use std::env;
use build_helper::sanitizer_lib_boilerplate;

use cmake::Config;

fn main() {
    if let Some(llvm_config) = env::var_os("LLVM_CONFIG") {
        build_helper::restore_library_path();

        let (native, target) = match sanitizer_lib_boilerplate("asan") {
            Ok(native) => native,
            _ => return,
        };

        Config::new(&native.src_dir)
            .define("COMPILER_RT_BUILD_SANITIZERS", "ON")
            .define("COMPILER_RT_BUILD_BUILTINS", "OFF")
            .define("COMPILER_RT_BUILD_XRAY", "OFF")
            .define("LLVM_CONFIG_PATH", llvm_config)
            .out_dir(&native.out_dir)
            .build_target(&target)
            .build();
        native.fixup_sanitizer_lib_name("asan");
    }
    println!("cargo:rerun-if-env-changed=LLVM_CONFIG");
}
