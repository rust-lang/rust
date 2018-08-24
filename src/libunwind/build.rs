use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target = env::var("TARGET").expect("TARGET was not set");

    if target.contains("linux") {
        if target.contains("musl") {
            // musl is handled in lib.rs
        } else if !target.contains("android") {
            println!("cargo:rustc-link-lib=gcc_s");
        }
    } else if target.contains("freebsd") {
        println!("cargo:rustc-link-lib=gcc_s");
    } else if target.contains("rumprun") {
        println!("cargo:rustc-link-lib=unwind");
    } else if target.contains("netbsd") {
        println!("cargo:rustc-link-lib=gcc_s");
    } else if target.contains("openbsd") {
        println!("cargo:rustc-link-lib=c++abi");
    } else if target.contains("solaris") {
        println!("cargo:rustc-link-lib=gcc_s");
    } else if target.contains("bitrig") {
        println!("cargo:rustc-link-lib=c++abi");
    } else if target.contains("dragonfly") {
        println!("cargo:rustc-link-lib=gcc_pic");
    } else if target.contains("windows-gnu") {
        println!("cargo:rustc-link-lib=static-nobundle=gcc_eh");
        println!("cargo:rustc-link-lib=static-nobundle=pthread");
    } else if target.contains("fuchsia") {
        println!("cargo:rustc-link-lib=unwind");
    } else if target.contains("haiku") {
        println!("cargo:rustc-link-lib=gcc_s");
    } else if target.contains("redox") {
        println!("cargo:rustc-link-lib=gcc");
    } else if target.contains("cloudabi") {
        println!("cargo:rustc-link-lib=unwind");
    }
}
