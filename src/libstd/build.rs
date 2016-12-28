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

extern crate gcc;
extern crate build_helper;

use std::env;
use std::path::PathBuf;
use std::process::Command;

use build_helper::run;

fn main() {
    println!("cargo:rustc-cfg=cargobuild");
    println!("cargo:rerun-if-changed=build.rs");

    let target = env::var("TARGET").expect("TARGET was not set");
    let host = env::var("HOST").expect("HOST was not set");
    if cfg!(feature = "backtrace") && !target.contains("apple") && !target.contains("msvc") &&
        !target.contains("emscripten") && !target.contains("fuchsia") && !target.contains("redox") {
        build_libbacktrace(&host, &target);
    }

    if target.contains("linux") {
        if target.contains("android") {
            println!("cargo:rustc-link-lib=dl");
            println!("cargo:rustc-link-lib=log");
            println!("cargo:rustc-link-lib=gcc");
        } else if !target.contains("musl") || target.contains("mips") {
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
    } else if target.contains("apple-darwin") {
        println!("cargo:rustc-link-lib=System");
    } else if target.contains("apple-ios") {
        println!("cargo:rustc-link-lib=System");
        println!("cargo:rustc-link-lib=objc");
        println!("cargo:rustc-link-lib=framework=Security");
        println!("cargo:rustc-link-lib=framework=Foundation");
    } else if target.contains("windows") {
        println!("cargo:rustc-link-lib=advapi32");
        println!("cargo:rustc-link-lib=ws2_32");
        println!("cargo:rustc-link-lib=userenv");
        println!("cargo:rustc-link-lib=shell32");
    } else if target.contains("fuchsia") {
        println!("cargo:rustc-link-lib=magenta");
        println!("cargo:rustc-link-lib=mxio");
        println!("cargo:rustc-link-lib=launchpad"); // for std::process
    }
}

fn build_libbacktrace(host: &str, target: &str) {
    let src_dir = env::current_dir().unwrap().join("../libbacktrace");
    let build_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());

    println!("cargo:rustc-link-lib=static=backtrace");
    println!("cargo:rustc-link-search=native={}/.libs", build_dir.display());

    let mut stack = src_dir.read_dir().unwrap()
                           .map(|e| e.unwrap())
                           .collect::<Vec<_>>();
    while let Some(entry) = stack.pop() {
        let path = entry.path();
        if entry.file_type().unwrap().is_dir() {
            stack.extend(path.read_dir().unwrap().map(|e| e.unwrap()));
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }

    let compiler = gcc::Config::new().get_compiler();
    // only msvc returns None for ar so unwrap is okay
    let ar = build_helper::cc2ar(compiler.path(), target).unwrap();
    let cflags = compiler.args().iter().map(|s| s.to_str().unwrap())
                         .collect::<Vec<_>>().join(" ");
    run(Command::new("sh")
                .current_dir(&build_dir)
                .arg(src_dir.join("configure").to_str().unwrap()
                            .replace("C:\\", "/c/")
                            .replace("\\", "/"))
                .arg("--with-pic")
                .arg("--disable-multilib")
                .arg("--disable-shared")
                .arg("--disable-host-shared")
                .arg(format!("--host={}", build_helper::gnu_target(target)))
                .arg(format!("--build={}", build_helper::gnu_target(host)))
                .env("CC", compiler.path())
                .env("AR", &ar)
                .env("RANLIB", format!("{} s", ar.display()))
                .env("CFLAGS", cflags));
    run(Command::new(build_helper::make(host))
                .current_dir(&build_dir)
                .arg(format!("INCDIR={}", src_dir.display()))
                .arg("-j").arg(env::var("NUM_JOBS").expect("NUM_JOBS was not set")));
}
