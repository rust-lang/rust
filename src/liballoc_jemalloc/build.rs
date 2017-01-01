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
extern crate gcc;

use std::env;
use std::path::PathBuf;
use std::process::Command;
use build_helper::run;

fn main() {
    println!("cargo:rustc-cfg=cargobuild");
    println!("cargo:rerun-if-changed=build.rs");

    let target = env::var("TARGET").expect("TARGET was not set");
    let host = env::var("HOST").expect("HOST was not set");
    let build_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let src_dir = env::current_dir().unwrap();

    // FIXME: This is a hack to support building targets that don't
    // support jemalloc alongside hosts that do. The jemalloc build is
    // controlled by a feature of the std crate, and if that feature
    // changes between targets, it invalidates the fingerprint of
    // std's build script (this is a cargo bug); so we must ensure
    // that the feature set used by std is the same across all
    // targets, which means we have to build the alloc_jemalloc crate
    // for targets like emscripten, even if we don't use it.
    if target.contains("rumprun") || target.contains("bitrig") || target.contains("openbsd") ||
       target.contains("msvc") || target.contains("emscripten") || target.contains("fuchsia") ||
       target.contains("redox") {
        println!("cargo:rustc-cfg=dummy_jemalloc");
        return;
    }

    if let Some(jemalloc) = env::var_os("JEMALLOC_OVERRIDE") {
        let jemalloc = PathBuf::from(jemalloc);
        println!("cargo:rustc-link-search=native={}",
                 jemalloc.parent().unwrap().display());
        let stem = jemalloc.file_stem().unwrap().to_str().unwrap();
        let name = jemalloc.file_name().unwrap().to_str().unwrap();
        let kind = if name.ends_with(".a") {
            "static"
        } else {
            "dylib"
        };
        println!("cargo:rustc-link-lib={}={}", kind, &stem[3..]);
        return;
    }

    let compiler = gcc::Config::new().get_compiler();
    // only msvc returns None for ar so unwrap is okay
    let ar = build_helper::cc2ar(compiler.path(), &target).unwrap();
    let cflags = compiler.args()
        .iter()
        .map(|s| s.to_str().unwrap())
        .collect::<Vec<_>>()
        .join(" ");

    let mut stack = src_dir.join("../jemalloc")
        .read_dir()
        .unwrap()
        .map(|e| e.unwrap())
        .filter(|e| &*e.file_name() != ".git")
        .collect::<Vec<_>>();
    while let Some(entry) = stack.pop() {
        let path = entry.path();
        if entry.file_type().unwrap().is_dir() {
            stack.extend(path.read_dir().unwrap().map(|e| e.unwrap()));
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }

    let mut cmd = Command::new("sh");
    cmd.arg(src_dir.join("../jemalloc/configure")
                   .to_str()
                   .unwrap()
                   .replace("C:\\", "/c/")
                   .replace("\\", "/"))
       .current_dir(&build_dir)
       .env("CC", compiler.path())
       .env("EXTRA_CFLAGS", cflags.clone())
       // jemalloc generates Makefile deps using GCC's "-MM" flag. This means
       // that GCC will run the preprocessor, and only the preprocessor, over
       // jemalloc's source files. If we don't specify CPPFLAGS, then at least
       // on ARM that step fails with a "Missing implementation for 32-bit
       // atomic operations" error. This is because no "-march" flag will be
       // passed to GCC, and then GCC won't define the
       // "__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4" macro that jemalloc needs to
       // select an atomic operation implementation.
       .env("CPPFLAGS", cflags.clone())
       .env("AR", &ar)
       .env("RANLIB", format!("{} s", ar.display()));

    if target.contains("windows") {
        // A bit of history here, this used to be --enable-lazy-lock added in
        // #14006 which was filed with jemalloc in jemalloc/jemalloc#83 which
        // was also reported to MinGW:
        //
        //  http://sourceforge.net/p/mingw-w64/bugs/395/
        //
        // When updating jemalloc to 4.0, however, it was found that binaries
        // would exit with the status code STATUS_RESOURCE_NOT_OWNED indicating
        // that a thread was unlocking a mutex it never locked. Disabling this
        // "lazy lock" option seems to fix the issue, but it was enabled by
        // default for MinGW targets in 13473c7 for jemalloc.
        //
        // As a result of all that, force disabling lazy lock on Windows, and
        // after reading some code it at least *appears* that the initialization
        // of mutexes is otherwise ok in jemalloc, so shouldn't cause problems
        // hopefully...
        //
        // tl;dr: make windows behave like other platforms by disabling lazy
        //        locking, but requires passing an option due to a historical
        //        default with jemalloc.
        cmd.arg("--disable-lazy-lock");
    } else if target.contains("ios") {
        cmd.arg("--disable-tls");
    } else if target.contains("android") {
        // We force android to have prefixed symbols because apparently
        // replacement of the libc allocator doesn't quite work. When this was
        // tested (unprefixed symbols), it was found that the `realpath`
        // function in libc would allocate with libc malloc (not jemalloc
        // malloc), and then the standard library would free with jemalloc free,
        // causing a segfault.
        //
        // If the test suite passes, however, without symbol prefixes then we
        // should be good to go!
        cmd.arg("--with-jemalloc-prefix=je_");
        cmd.arg("--disable-tls");
    } else if target.contains("dragonfly") {
        cmd.arg("--with-jemalloc-prefix=je_");
    }

    if cfg!(feature = "debug-jemalloc") {
        cmd.arg("--enable-debug");
    }

    // Turn off broken quarantine (see jemalloc/jemalloc#161)
    cmd.arg("--disable-fill");
    cmd.arg(format!("--host={}", build_helper::gnu_target(&target)));
    cmd.arg(format!("--build={}", build_helper::gnu_target(&host)));

    // for some reason, jemalloc configure doesn't detect this value
    // automatically for this target
    if target == "sparc64-unknown-linux-gnu" {
        cmd.arg("--with-lg-quantum=4");
    }

    run(&mut cmd);
    let mut make = Command::new(build_helper::make(&host));
    make.current_dir(&build_dir)
        .arg("build_lib_static");

    // mingw make seems... buggy? unclear...
    if !host.contains("windows") {
        make.arg("-j")
            .arg(env::var("NUM_JOBS").expect("NUM_JOBS was not set"));
    }

    run(&mut make);

    if target.contains("windows") {
        println!("cargo:rustc-link-lib=static=jemalloc");
    } else {
        println!("cargo:rustc-link-lib=static=jemalloc_pic");
    }
    println!("cargo:rustc-link-search=native={}/lib", build_dir.display());
    if target.contains("android") {
        println!("cargo:rustc-link-lib=gcc");
    } else if !target.contains("windows") && !target.contains("musl") {
        println!("cargo:rustc-link-lib=pthread");
    }
}
