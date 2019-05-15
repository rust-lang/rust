#![deny(warnings)]

use std::env;

fn main() {
    let target = env::var("TARGET").expect("TARGET was not set");
    if target.contains("linux") {
        if target.contains("android") {
            println!("cargo:rustc-link-lib=dl");
            println!("cargo:rustc-link-lib=log");
            println!("cargo:rustc-link-lib=gcc");
        } else if !target.contains("musl") {
            println!("cargo:rustc-link-lib=dl");
            println!("cargo:rustc-link-lib=rt");
            println!("cargo:rustc-link-lib=pthread");
        }
    } else if target.contains("freebsd") {
        println!("cargo:rustc-link-lib=execinfo");
        println!("cargo:rustc-link-lib=pthread");
    } else if target.contains("netbsd") {
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=rt");
    } else if target.contains("dragonfly") || target.contains("openbsd") {
        println!("cargo:rustc-link-lib=pthread");
    } else if target.contains("solaris") {
        println!("cargo:rustc-link-lib=socket");
        println!("cargo:rustc-link-lib=posix4");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=resolv");
    } else if target.contains("apple-darwin") {
        println!("cargo:rustc-link-lib=System");

        // res_init and friends require -lresolv on macOS/iOS.
        // See #41582 and http://blog.achernya.com/2013/03/os-x-has-silly-libsystem.html
        println!("cargo:rustc-link-lib=resolv");
    } else if target.contains("apple-ios") {
        println!("cargo:rustc-link-lib=System");
        println!("cargo:rustc-link-lib=objc");
        println!("cargo:rustc-link-lib=framework=Security");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=resolv");
    } else if target.contains("windows") {
        println!("cargo:rustc-link-lib=advapi32");
        println!("cargo:rustc-link-lib=ws2_32");
        println!("cargo:rustc-link-lib=userenv");
    } else if target.contains("fuchsia") {
        println!("cargo:rustc-link-lib=zircon");
        println!("cargo:rustc-link-lib=fdio");
    } else if target.contains("cloudabi") {
        if cfg!(feature = "backtrace") {
            println!("cargo:rustc-link-lib=unwind");
        }
        println!("cargo:rustc-link-lib=c");
        println!("cargo:rustc-link-lib=compiler_rt");
    }
}
