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
extern crate cc;

use build_helper::native_lib_boilerplate;
use std::env;
use std::fs::File;

fn main() {
    let target = env::var("TARGET").expect("TARGET was not set");
    if cfg!(feature = "backtrace") &&
        !target.contains("cloudabi") &&
        !target.contains("emscripten") &&
        !target.contains("msvc") &&
        !target.contains("wasm32")
    {
        let _ = build_libbacktrace(&target);
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
    } else if target.contains("netbsd") {
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=rt");
    } else if target.contains("dragonfly") || target.contains("bitrig") ||
              target.contains("openbsd") {
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

fn build_libbacktrace(target: &str) -> Result<(), ()> {
    let native = native_lib_boilerplate("libbacktrace", "libbacktrace", "backtrace", "")?;

    let mut build = cc::Build::new();
    build
        .flag("-fvisibility=hidden")
        .include("../libbacktrace")
        .include(&native.out_dir)
        .out_dir(&native.out_dir)
        .warnings(false)
        .file("../libbacktrace/alloc.c")
        .file("../libbacktrace/backtrace.c")
        .file("../libbacktrace/dwarf.c")
        .file("../libbacktrace/fileline.c")
        .file("../libbacktrace/posix.c")
        .file("../libbacktrace/read.c")
        .file("../libbacktrace/sort.c")
        .file("../libbacktrace/state.c");

    let any_debug = env::var("RUSTC_DEBUGINFO").unwrap_or_default() == "true" ||
        env::var("RUSTC_DEBUGINFO_LINES").unwrap_or_default() == "true";
    build.debug(any_debug);

    if target.contains("darwin") {
        build.file("../libbacktrace/macho.c");
    } else if target.contains("windows") {
        build.file("../libbacktrace/pecoff.c");
    } else {
        build.file("../libbacktrace/elf.c");

        let pointer_width = env::var("CARGO_CFG_TARGET_POINTER_WIDTH").unwrap();
        if pointer_width == "64" {
            build.define("BACKTRACE_ELF_SIZE", "64");
        } else {
            build.define("BACKTRACE_ELF_SIZE", "32");
        }
    }

    File::create(native.out_dir.join("backtrace-supported.h")).unwrap();
    build.define("BACKTRACE_SUPPORTED", "1");
    build.define("BACKTRACE_USES_MALLOC", "1");
    build.define("BACKTRACE_SUPPORTS_THREADS", "0");
    build.define("BACKTRACE_SUPPORTS_DATA", "0");

    File::create(native.out_dir.join("config.h")).unwrap();
    if !target.contains("apple-ios") &&
       !target.contains("solaris") &&
       !target.contains("redox") &&
       !target.contains("android") &&
       !target.contains("haiku") {
        build.define("HAVE_DL_ITERATE_PHDR", "1");
    }
    build.define("_GNU_SOURCE", "1");
    build.define("_LARGE_FILES", "1");

    build.compile("backtrace");
    Ok(())
}
