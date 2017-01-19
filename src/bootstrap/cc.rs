// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! C-compiler probing and detection.
//!
//! This module will fill out the `cc` and `cxx` maps of `Build` by looking for
//! C and C++ compilers for each target configured. A compiler is found through
//! a number of vectors (in order of precedence)
//!
//! 1. Configuration via `target.$target.cc` in `config.toml`.
//! 2. Configuration via `target.$target.android-ndk` in `config.toml`, if
//!    applicable
//! 3. Special logic to probe on OpenBSD
//! 4. The `CC_$target` environment variable.
//! 5. The `CC` environment variable.
//! 6. "cc"
//!
//! Some of this logic is implemented here, but much of it is farmed out to the
//! `gcc` crate itself, so we end up having the same fallbacks as there.
//! Similar logic is then used to find a C++ compiler, just some s/cc/c++/ is
//! used.
//!
//! It is intended that after this module has run no C/C++ compiler will
//! ever be probed for. Instead the compilers found here will be used for
//! everything.

use std::process::Command;

use build_helper::{cc2ar, output};
use gcc;

use Build;
use config::Target;

pub fn find(build: &mut Build) {
    // For all targets we're going to need a C compiler for building some shims
    // and such as well as for being a linker for Rust code.
    for target in build.config.target.iter() {
        let mut cfg = gcc::Config::new();
        cfg.cargo_metadata(false).opt_level(0).debug(false)
           .target(target).host(&build.config.build);

        let config = build.config.target_config.get(target);
        if let Some(cc) = config.and_then(|c| c.cc.as_ref()) {
            cfg.compiler(cc);
        } else {
            set_compiler(&mut cfg, "gcc", target, config, build);
        }

        let compiler = cfg.get_compiler();
        let ar = cc2ar(compiler.path(), target);
        build.verbose(&format!("CC_{} = {:?}", target, compiler.path()));
        if let Some(ref ar) = ar {
            build.verbose(&format!("AR_{} = {:?}", target, ar));
        }
        build.cc.insert(target.to_string(), (compiler, ar));
    }

    // For all host triples we need to find a C++ compiler as well
    for host in build.config.host.iter() {
        let mut cfg = gcc::Config::new();
        cfg.cargo_metadata(false).opt_level(0).debug(false).cpp(true)
           .target(host).host(&build.config.build);
        let config = build.config.target_config.get(host);
        if let Some(cxx) = config.and_then(|c| c.cxx.as_ref()) {
            cfg.compiler(cxx);
        } else {
            set_compiler(&mut cfg, "g++", host, config, build);
        }
        let compiler = cfg.get_compiler();
        build.verbose(&format!("CXX_{} = {:?}", host, compiler.path()));
        build.cxx.insert(host.to_string(), compiler);
    }
}

fn set_compiler(cfg: &mut gcc::Config,
                gnu_compiler: &str,
                target: &str,
                config: Option<&Target>,
                build: &Build) {
    match target {
        // When compiling for android we may have the NDK configured in the
        // config.toml in which case we look there. Otherwise the default
        // compiler already takes into account the triple in question.
        t if t.contains("android") => {
            if let Some(ndk) = config.and_then(|c| c.ndk.as_ref()) {
                let target = target.replace("armv7", "arm");
                let compiler = format!("{}-{}", target, gnu_compiler);
                cfg.compiler(ndk.join("bin").join(compiler));
            }
        }

        // The default gcc version from OpenBSD may be too old, try using egcc,
        // which is a gcc version from ports, if this is the case.
        t if t.contains("openbsd") => {
            let c = cfg.get_compiler();
            if !c.path().ends_with(gnu_compiler) {
                return
            }

            let output = output(c.to_command().arg("--version"));
            let i = match output.find(" 4.") {
                Some(i) => i,
                None => return,
            };
            match output[i + 3..].chars().next().unwrap() {
                '0' ... '6' => {}
                _ => return,
            }
            let alternative = format!("e{}", gnu_compiler);
            if Command::new(&alternative).output().is_ok() {
                cfg.compiler(alternative);
            }
        }

        "mips-unknown-linux-musl" => {
            if cfg.get_compiler().path().to_str() == Some("gcc") {
                cfg.compiler("mips-linux-musl-gcc");
            }
        }
        "mipsel-unknown-linux-musl" => {
            if cfg.get_compiler().path().to_str() == Some("gcc") {
                cfg.compiler("mipsel-linux-musl-gcc");
            }
        }

        t if t.contains("musl") => {
            if let Some(root) = build.musl_root(target) {
                let guess = root.join("bin/musl-gcc");
                if guess.exists() {
                    cfg.compiler(guess);
                }
            }
        }

        _ => {}
    }
}
