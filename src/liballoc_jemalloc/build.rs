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

use std::env;
use std::path::PathBuf;
use std::process::Command;
use build_helper::{run, native_lib_boilerplate, BuildExpectation};

fn main() {
    // FIXME: This is a hack to support building targets that don't
    // support jemalloc alongside hosts that do. The jemalloc build is
    // controlled by a feature of the std crate, and if that feature
    // changes between targets, it invalidates the fingerprint of
    // std's build script (this is a cargo bug); so we must ensure
    // that the feature set used by std is the same across all
    // targets, which means we have to build the alloc_jemalloc crate
    // for targets like emscripten, even if we don't use it.
    let target = env::var("TARGET").expect("TARGET was not set");
    let host = env::var("HOST").expect("HOST was not set");
    if target.contains("rumprun") || target.contains("bitrig") || target.contains("openbsd") ||
       target.contains("msvc") || target.contains("emscripten") || target.contains("fuchsia") ||
       target.contains("redox") {
        println!("cargo:rustc-cfg=dummy_jemalloc");
        return;
    }

    if target.contains("android") {
        println!("cargo:rustc-link-lib=gcc");
    } else if !target.contains("windows") && !target.contains("musl") {
        println!("cargo:rustc-link-lib=pthread");
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

    let link_name = if target.contains("windows") { "jemalloc" } else { "jemalloc_pic" };
    let native = match native_lib_boilerplate("jemalloc", "jemalloc", link_name, "lib") {
        Ok(native) => native,
        _ => return,
    };

    let compiler = cc::Build::new().get_compiler();
    // only msvc returns None for ar so unwrap is okay
    let ar = build_helper::cc2ar(compiler.path(), &target).unwrap();
    let cflags = compiler.args()
        .iter()
        .map(|s| s.to_str().unwrap())
        .filter(|&s| {
            // separate function/data sections trigger errors with android's TLS emulation
            !target.contains("android") || (s != "-ffunction-sections" && s != "-fdata-sections")
        })
        .collect::<Vec<_>>()
        .join(" ");

    let mut cmd = Command::new("sh");
    cmd.arg(native.src_dir.join("configure")
                          .to_str()
                          .unwrap()
                          .replace("C:\\", "/c/")
                          .replace("\\", "/"))
       .arg("--disable-cxx")
       .current_dir(&native.out_dir)
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

    if target.contains("android") {
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
    } else if target.contains("dragonfly") || target.contains("musl") {
        cmd.arg("--with-jemalloc-prefix=je_");
    }

    // FIXME: building with jemalloc assertions is currently broken.
    // See <https://github.com/rust-lang/rust/issues/44152>.
    //if cfg!(feature = "debug") {
    //    cmd.arg("--enable-debug");
    //}

    cmd.arg(format!("--host={}", build_helper::gnu_target(&target)));
    cmd.arg(format!("--build={}", build_helper::gnu_target(&host)));

    // for some reason, jemalloc configure doesn't detect this value
    // automatically for this target
    if target == "sparc64-unknown-linux-gnu" {
        cmd.arg("--with-lg-quantum=4");
    }

    run(&mut cmd, BuildExpectation::None);

    let mut make = Command::new(build_helper::make(&host));
    make.current_dir(&native.out_dir)
        .arg("build_lib_static");

    // These are intended for mingw32-make which we don't use
    if cfg!(windows) {
        make.env_remove("MAKEFLAGS").env_remove("MFLAGS");
    }

    // mingw make seems... buggy? unclear...
    if !host.contains("windows") {
        make.arg("-j")
            .arg(env::var("NUM_JOBS").expect("NUM_JOBS was not set"));
    }

    run(&mut make, BuildExpectation::None);

    // The pthread_atfork symbols is used by jemalloc on android but the really
    // old android we're building on doesn't have them defined, so just make
    // sure the symbols are available.
    if target.contains("androideabi") {
        println!("cargo:rerun-if-changed=pthread_atfork_dummy.c");
        cc::Build::new()
            .flag("-fvisibility=hidden")
            .file("pthread_atfork_dummy.c")
            .compile("libpthread_atfork_dummy.a");
    }
}
