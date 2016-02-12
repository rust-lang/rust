// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate bootstrap;

use std::env;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let args = env::args_os().skip(1).collect::<Vec<_>>();
    // Detect whether or not we're a build script depending on whether --target
    // is passed (a bit janky...)
    let is_build_script = args.iter()
                              .position(|i| i.to_str() == Some("--target"))
                              .is_none();

    // Build scripts always use the snapshot compiler which is guaranteed to be
    // able to produce an executable, whereas intermediate compilers may not
    // have the standard library built yet and may not be able to produce an
    // executable. Otherwise we just use the standard compiler we're
    // bootstrapping with.
    let rustc = if is_build_script {
        env::var_os("RUSTC_SNAPSHOT").unwrap()
    } else {
        env::var_os("RUSTC_REAL").unwrap()
    };

    let mut cmd = Command::new(rustc);
    cmd.args(&args)
       .arg("--cfg").arg(format!("stage{}", env::var("RUSTC_STAGE").unwrap()));

    if is_build_script {
        // Build scripts are always built with the snapshot compiler, so we need
        // to be sure to set up the right path information for the OS dynamic
        // linker to find the libraries in question.
        if let Some(p) = env::var_os("RUSTC_SNAPSHOT_LIBDIR") {
            let mut path = bootstrap::dylib_path();
            path.insert(0, PathBuf::from(p));
            cmd.env(bootstrap::dylib_path_var(), env::join_paths(path).unwrap());
        }
    } else {
        cmd.arg("--sysroot").arg(env::var_os("RUSTC_SYSROOT").unwrap());

        // When we build Rust dylibs they're all intended for intermediate
        // usage, so make sure we pass the -Cprefer-dynamic flag instead of
        // linking all deps statically into the dylib.
        cmd.arg("-Cprefer-dynamic");

        if let Some(s) = env::var_os("MUSL_ROOT") {
            let mut root = OsString::from("native=");
            root.push(&s);
            root.push("/lib");
            cmd.arg("-L").arg(&root);
        }
    }

    // Set various options from config.toml to configure how we're building
    // code.
    if env::var("RUSTC_DEBUGINFO") == Ok("true".to_string()) {
        cmd.arg("-g");
    }
    if env::var("RUSTC_RPATH") == Ok("true".to_string()) {
        cmd.arg("-Crpath");
    }
    let debug_assertions = match env::var("RUSTC_DEBUG_ASSERTIONS") {
        Ok(s) => if s == "true" {"y"} else {"n"},
        Err(..) => "n",
    };
    cmd.arg("-C").arg(format!("debug-assertions={}", debug_assertions));
    if let Ok(s) = env::var("RUSTC_CODEGEN_UNITS") {
        cmd.arg("-C").arg(format!("codegen-units={}", s));
    }
    if let Ok(s) = env::var("RUSTC_FLAGS") {
        cmd.args(&s.split(" ").filter(|s| !s.is_empty()).collect::<Vec<_>>());
    }

    // Actually run the compiler!
    std::process::exit(match cmd.status() {
        Ok(s) => s.code().unwrap_or(1),
        Err(e) => panic!("\n\nfailed to run {:?}: {}\n\n", cmd, e),
    })
}
