use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target = env::var("TARGET").expect("TARGET was not set");

    if target.contains("android") {
        let build = cc::Build::new();

        // Since ndk r23 beta 3 `libgcc` was replaced with `libunwind` thus
        // check if we have `libunwind` available and if so use it. Otherwise
        // fall back to `libgcc` to support older ndk versions.
        let has_unwind = build.is_flag_supported("-lunwind").expect("Unable to invoke compiler");

        if has_unwind {
            println!("cargo:rustc-link-lib=unwind");
        } else {
            println!("cargo:rustc-link-lib=gcc");
        }
    } else if target.contains("freebsd") {
        println!("cargo:rustc-link-lib=gcc_s");
    } else if target.contains("netbsd") {
        println!("cargo:rustc-link-lib=gcc_s");
    } else if target.contains("openbsd") {
        if target.contains("sparc64") {
            println!("cargo:rustc-link-lib=gcc");
        } else {
            println!("cargo:rustc-link-lib=c++abi");
        }
    } else if target.contains("solaris") {
        println!("cargo:rustc-link-lib=gcc_s");
    } else if target.contains("illumos") {
        println!("cargo:rustc-link-lib=gcc_s");
    } else if target.contains("dragonfly") {
        println!("cargo:rustc-link-lib=gcc_pic");
    } else if target.contains("pc-windows-gnu") {
        // This is handled in the target spec with late_link_args_[static|dynamic]
    } else if target.contains("uwp-windows-gnu") {
        println!("cargo:rustc-link-lib=unwind");
    } else if target.contains("fuchsia") {
        println!("cargo:rustc-link-lib=unwind");
    } else if target.contains("haiku") {
        println!("cargo:rustc-link-lib=gcc_s");
    } else if target.contains("redox") {
        // redox is handled in lib.rs
    }
}
