// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(warnings)]

extern crate build_helper;

use std::env;
use std::process::Command;
use build_helper::{run, native_lib_boilerplate};

fn main() {
    let target = env::var("TARGET").expect("TARGET was not set");
    let host = env::var("HOST").expect("HOST was not set");
    if cfg!(feature = "backtrace") &&
        !target.contains("cloudabi") &&
        !target.contains("emscripten") &&
        !target.contains("fuchsia") &&
        !target.contains("msvc") &&
        !target.contains("wasm32")
    {
        let _ = build_libbacktrace(&host, &target);
    }

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
    } else if target.contains("dragonfly") || target.contains("bitrig") ||
              target.contains("netbsd") || target.contains("openbsd") {
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
        println!("cargo:rustc-link-lib=shell32");
    } else if target.contains("fuchsia") {
        // use system-provided libbacktrace
        if cfg!(feature = "backtrace") {
            println!("cargo:rustc-link-lib=backtrace");
        }
        println!("cargo:rustc-link-lib=zircon");
        println!("cargo:rustc-link-lib=fdio");
        println!("cargo:rustc-link-lib=launchpad"); // for std::process
    } else if target.contains("cloudabi") {
        if cfg!(feature = "backtrace") {
            println!("cargo:rustc-link-lib=unwind");
        }
        println!("cargo:rustc-link-lib=c");
        println!("cargo:rustc-link-lib=compiler_rt");
    }
}

fn build_libbacktrace(host: &str, target: &str) -> Result<(), ()> {
    let native = native_lib_boilerplate("libbacktrace", "libbacktrace", "backtrace", ".libs")?;

    run(Command::new("sh")
                .current_dir(&native.out_dir)
                .arg(native.src_dir.join("configure").to_str().unwrap()
                                   .replace("C:\\", "/c/")
                                   .replace("\\", "/"))
                .arg("--with-pic")
                .arg("--disable-multilib")
                .arg("--disable-shared")
                .arg("--disable-host-shared")
                .arg(format!("--host={}", build_helper::gnu_target(target)))
                .arg(format!("--build={}", build_helper::gnu_target(host)))
                .env("CFLAGS", env::var("CFLAGS").unwrap_or_default() + " -fvisibility=hidden"));

    run(Command::new(build_helper::make(host))
                .current_dir(&native.out_dir)
                .arg(format!("INCDIR={}", native.src_dir.display()))
                .arg("-j").arg(env::var("NUM_JOBS").expect("NUM_JOBS was not set")));
    Ok(())
}
