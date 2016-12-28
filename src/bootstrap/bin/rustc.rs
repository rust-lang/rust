// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Shim which is passed to Cargo as "rustc" when running the bootstrap.
//!
//! This shim will take care of some various tasks that our build process
//! requires that Cargo can't quite do through normal configuration:
//!
//! 1. When compiling build scripts and build dependencies, we need a guaranteed
//!    full standard library available. The only compiler which actually has
//!    this is the snapshot, so we detect this situation and always compile with
//!    the snapshot compiler.
//! 2. We pass a bunch of `--cfg` and other flags based on what we're compiling
//!    (and this slightly differs based on a whether we're using a snapshot or
//!    not), so we do that all here.
//!
//! This may one day be replaced by RUSTFLAGS, but the dynamic nature of
//! switching compilers for the bootstrap and for build scripts will probably
//! never get replaced.

#![deny(warnings)]

extern crate bootstrap;

use std::env;
use std::ffi::OsString;
use std::io;
use std::io::prelude::*;
use std::str::FromStr;
use std::path::PathBuf;
use std::process::{Command, ExitStatus};

fn main() {
    let args = env::args_os().skip(1).collect::<Vec<_>>();
    // Detect whether or not we're a build script depending on whether --target
    // is passed (a bit janky...)
    let target = args.windows(2)
        .find(|w| &*w[0] == "--target")
        .and_then(|w| w[1].to_str());
    let version = args.iter().find(|w| &**w == "-vV");

    let verbose = match env::var("RUSTC_VERBOSE") {
        Ok(s) => usize::from_str(&s).expect("RUSTC_VERBOSE should be an integer"),
        Err(_) => 0,
    };

    // Build scripts always use the snapshot compiler which is guaranteed to be
    // able to produce an executable, whereas intermediate compilers may not
    // have the standard library built yet and may not be able to produce an
    // executable. Otherwise we just use the standard compiler we're
    // bootstrapping with.
    //
    // Also note that cargo will detect the version of the compiler to trigger
    // a rebuild when the compiler changes. If this happens, we want to make
    // sure to use the actual compiler instead of the snapshot compiler becase
    // that's the one that's actually changing.
    let (rustc, libdir) = if target.is_none() && version.is_none() {
        ("RUSTC_SNAPSHOT", "RUSTC_SNAPSHOT_LIBDIR")
    } else {
        ("RUSTC_REAL", "RUSTC_LIBDIR")
    };
    let stage = env::var("RUSTC_STAGE").expect("RUSTC_STAGE was not set");
    let sysroot = env::var_os("RUSTC_SYSROOT").expect("RUSTC_SYSROOT was not set");

    let rustc = env::var_os(rustc).unwrap_or_else(|| panic!("{:?} was not set", rustc));
    let libdir = env::var_os(libdir).unwrap_or_else(|| panic!("{:?} was not set", libdir));
    let mut dylib_path = bootstrap::util::dylib_path();
    dylib_path.insert(0, PathBuf::from(libdir));

    let mut cmd = Command::new(rustc);
    cmd.args(&args)
        .arg("--cfg")
        .arg(format!("stage{}", stage))
        .env(bootstrap::util::dylib_path_var(),
             env::join_paths(&dylib_path).unwrap());

    if let Some(target) = target {
        // The stage0 compiler has a special sysroot distinct from what we
        // actually downloaded, so we just always pass the `--sysroot` option.
        cmd.arg("--sysroot").arg(sysroot);

        // When we build Rust dylibs they're all intended for intermediate
        // usage, so make sure we pass the -Cprefer-dynamic flag instead of
        // linking all deps statically into the dylib.
        if env::var_os("RUSTC_NO_PREFER_DYNAMIC").is_none() {
            cmd.arg("-Cprefer-dynamic");
        }

        // Help the libc crate compile by assisting it in finding the MUSL
        // native libraries.
        if let Some(s) = env::var_os("MUSL_ROOT") {
            let mut root = OsString::from("native=");
            root.push(&s);
            root.push("/lib");
            cmd.arg("-L").arg(&root);
        }

        // Pass down extra flags, commonly used to configure `-Clinker` when
        // cross compiling.
        if let Ok(s) = env::var("RUSTC_FLAGS") {
            cmd.args(&s.split(" ").filter(|s| !s.is_empty()).collect::<Vec<_>>());
        }

        // Pass down incremental directory, if any.
        if let Ok(dir) = env::var("RUSTC_INCREMENTAL") {
            cmd.arg(format!("-Zincremental={}", dir));

            if verbose > 0 {
                cmd.arg("-Zincremental-info");
            }
        }

        // If we're compiling specifically the `panic_abort` crate then we pass
        // the `-C panic=abort` option. Note that we do not do this for any
        // other crate intentionally as this is the only crate for now that we
        // ship with panic=abort.
        //
        // This... is a bit of a hack how we detect this. Ideally this
        // information should be encoded in the crate I guess? Would likely
        // require an RFC amendment to RFC 1513, however.
        let is_panic_abort = args.windows(2)
            .any(|a| &*a[0] == "--crate-name" && &*a[1] == "panic_abort");
        if is_panic_abort {
            cmd.arg("-C").arg("panic=abort");
        }

        // Set various options from config.toml to configure how we're building
        // code.
        if env::var("RUSTC_DEBUGINFO") == Ok("true".to_string()) {
            cmd.arg("-g");
        } else if env::var("RUSTC_DEBUGINFO_LINES") == Ok("true".to_string()) {
            cmd.arg("-Cdebuginfo=1");
        }
        let debug_assertions = match env::var("RUSTC_DEBUG_ASSERTIONS") {
            Ok(s) => if s == "true" { "y" } else { "n" },
            Err(..) => "n",
        };
        cmd.arg("-C").arg(format!("debug-assertions={}", debug_assertions));
        if let Ok(s) = env::var("RUSTC_CODEGEN_UNITS") {
            cmd.arg("-C").arg(format!("codegen-units={}", s));
        }

        // Emit save-analysis info.
        if env::var("RUSTC_SAVE_ANALYSIS") == Ok("api".to_string()) {
            cmd.arg("-Zsave-analysis-api");
        }

        // Dealing with rpath here is a little special, so let's go into some
        // detail. First off, `-rpath` is a linker option on Unix platforms
        // which adds to the runtime dynamic loader path when looking for
        // dynamic libraries. We use this by default on Unix platforms to ensure
        // that our nightlies behave the same on Windows, that is they work out
        // of the box. This can be disabled, of course, but basically that's why
        // we're gated on RUSTC_RPATH here.
        //
        // Ok, so the astute might be wondering "why isn't `-C rpath` used
        // here?" and that is indeed a good question to task. This codegen
        // option is the compiler's current interface to generating an rpath.
        // Unfortunately it doesn't quite suffice for us. The flag currently
        // takes no value as an argument, so the compiler calculates what it
        // should pass to the linker as `-rpath`. This unfortunately is based on
        // the **compile time** directory structure which when building with
        // Cargo will be very different than the runtime directory structure.
        //
        // All that's a really long winded way of saying that if we use
        // `-Crpath` then the executables generated have the wrong rpath of
        // something like `$ORIGIN/deps` when in fact the way we distribute
        // rustc requires the rpath to be `$ORIGIN/../lib`.
        //
        // So, all in all, to set up the correct rpath we pass the linker
        // argument manually via `-C link-args=-Wl,-rpath,...`. Plus isn't it
        // fun to pass a flag to a tool to pass a flag to pass a flag to a tool
        // to change a flag in a binary?
        if env::var("RUSTC_RPATH") == Ok("true".to_string()) {
            let rpath = if target.contains("apple") {

                // Note that we need to take one extra step on OSX to also pass
                // `-Wl,-instal_name,@rpath/...` to get things to work right. To
                // do that we pass a weird flag to the compiler to get it to do
                // so. Note that this is definitely a hack, and we should likely
                // flesh out rpath support more fully in the future.
                if stage != "0" {
                    cmd.arg("-Z").arg("osx-rpath-install-name");
                }
                Some("-Wl,-rpath,@loader_path/../lib")
            } else if !target.contains("windows") {
                Some("-Wl,-rpath,$ORIGIN/../lib")
            } else {
                None
            };
            if let Some(rpath) = rpath {
                cmd.arg("-C").arg(format!("link-args={}", rpath));
            }

            if let Ok(s) = env::var("RUSTFLAGS") {
                for flag in s.split_whitespace() {
                    cmd.arg(flag);
                }
            }
        }
    }

    if verbose > 1 {
        writeln!(&mut io::stderr(), "rustc command: {:?}", cmd).unwrap();
    }

    // Actually run the compiler!
    std::process::exit(match exec_cmd(&mut cmd) {
        Ok(s) => s.code().unwrap_or(0xfe),
        Err(e) => panic!("\n\nfailed to run {:?}: {}\n\n", cmd, e),
    })
}

#[cfg(unix)]
fn exec_cmd(cmd: &mut Command) -> ::std::io::Result<ExitStatus> {
    use std::os::unix::process::CommandExt;
    Err(cmd.exec())
}

#[cfg(not(unix))]
fn exec_cmd(cmd: &mut Command) -> ::std::io::Result<ExitStatus> {
    cmd.status()
}
