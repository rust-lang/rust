// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::env;

fn main() {
    println!("cargo:rustc-cfg=cargobuild");

    let target = env::var("TARGET").expect("TARGET was not set");

    if target.contains("linux") {
        if target.contains("musl") && !target.contains("mips") {
            println!("cargo:rustc-link-lib=static=unwind");
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
        println!("cargo:rustc-link-lib=gcc");
    } else if target.contains("bitrig") {
        println!("cargo:rustc-link-lib=c++abi");
    } else if target.contains("dragonfly") {
        println!("cargo:rustc-link-lib=gcc_pic");
    } else if target.contains("windows-gnu") {
        println!("cargo:rustc-link-lib=gcc_eh");
    } else if target.contains("fuchsia") {
        println!("cargo:rustc-link-lib=unwind");
    }
}
