use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_MIRI");

    if env::var_os("CARGO_CFG_MIRI").is_some() {
        // Miri doesn't need the linker flags or a libunwind build.
        return;
    }

    let target = env::var("TARGET").expect("TARGET was not set");
    if target.contains("android") {
        let build = cc::Build::new();

        // Since ndk r23 beta 3 `libgcc` was replaced with `libunwind` thus
        // check if we have `libunwind` available and if so use it. Otherwise
        // fall back to `libgcc` to support older ndk versions.
        let has_unwind = build.is_flag_supported("-lunwind").expect("Unable to invoke compiler");

        if has_unwind {
            println!("cargo:rustc-cfg=feature=\"system-llvm-libunwind\"");
        }
    }
}
