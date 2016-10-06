// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of the install aspects of the compiler.
//!
//! This module is responsible for installing the standard library,
//! compiler, and documentation.

use std::fs;
use std::borrow::Cow;
use std::path::Path;
use std::process::Command;

use Build;
use dist::{package_vers, sanitize_sh, tmpdir};

/// Installs everything.
pub fn install(build: &Build, stage: u32, host: &str) {
    let prefix = build.config.prefix.as_ref().clone().map(|x| Path::new(x))
        .unwrap_or(Path::new("/usr/local"));
    let docdir = build.config.docdir.as_ref().clone().map(|x| Cow::Borrowed(Path::new(x)))
        .unwrap_or(Cow::Owned(prefix.join("share/doc/rust")));
    let libdir = build.config.libdir.as_ref().clone().map(|x| Cow::Borrowed(Path::new(x)))
        .unwrap_or(Cow::Owned(prefix.join("lib")));
    let mandir = build.config.mandir.as_ref().clone().map(|x| Cow::Borrowed(Path::new(x)))
        .unwrap_or(Cow::Owned(prefix.join("share/man")));
    let empty_dir = build.out.join("tmp/empty_dir");
    t!(fs::create_dir_all(&empty_dir));
    if build.config.docs {
        install_sh(&build, "docs", "rust-docs", stage, host, prefix,
                   &docdir, &libdir, &mandir, &empty_dir);
    }
    install_sh(&build, "std", "rust-std", stage, host, prefix,
               &docdir, &libdir, &mandir, &empty_dir);
    install_sh(&build, "rustc", "rustc", stage, host, prefix,
               &docdir, &libdir, &mandir, &empty_dir);
    t!(fs::remove_dir_all(&empty_dir));
}

fn install_sh(build: &Build, package: &str, name: &str, stage: u32, host: &str,
              prefix: &Path, docdir: &Path, libdir: &Path, mandir: &Path, empty_dir: &Path) {
    println!("Install {} stage{} ({})", package, stage, host);
    let package_name = format!("{}-{}-{}", name, package_vers(build), host);

    let mut cmd = Command::new("sh");
    cmd.current_dir(empty_dir)
       .arg(sanitize_sh(&tmpdir(build).join(&package_name).join("install.sh")))
       .arg(format!("--prefix={}", sanitize_sh(prefix)))
       .arg(format!("--docdir={}", sanitize_sh(docdir)))
       .arg(format!("--libdir={}", sanitize_sh(libdir)))
       .arg(format!("--mandir={}", sanitize_sh(mandir)))
       .arg("--disable-ldconfig");
    build.run(&mut cmd);
}
