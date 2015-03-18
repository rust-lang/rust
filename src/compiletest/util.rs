// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::env;
use common::Config;

/// Conversion table from triple OS name to Rust SYSNAME
const OS_TABLE: &'static [(&'static str, &'static str)] = &[
    ("mingw32", "windows"),
    ("win32", "windows"),
    ("windows", "windows"),
    ("darwin", "macos"),
    ("android", "android"),
    ("linux", "linux"),
    ("freebsd", "freebsd"),
    ("dragonfly", "dragonfly"),
    ("bitrig", "bitrig"),
    ("openbsd", "openbsd"),
];

const ARCH_TABLE: &'static [(&'static str, &'static str)] = &[
    ("i386", "x86"),
    ("i686", "x86"),
    ("amd64", "x86_64"),
    ("x86_64", "x86_64"),
    ("sparc", "sparc"),
    ("powerpc", "powerpc"),
    ("arm64", "aarch64"),
    ("arm", "arm"),
    ("aarch64", "aarch64"),
    ("mips", "mips"),
    ("xcore", "xcore"),
    ("msp430", "msp430"),
    ("hexagon", "hexagon"),
    ("s390x", "systemz"),
];

pub fn get_os(triple: &str) -> &'static str {
    for &(triple_os, os) in OS_TABLE {
        if triple.contains(triple_os) {
            return os
        }
    }
    panic!("Cannot determine OS from triple");
}
pub fn get_arch(triple: &str) -> &'static str {
    for &(triple_arch, arch) in ARCH_TABLE {
        if triple.contains(triple_arch) {
            return arch
        }
    }
    panic!("Cannot determine Architecture from triple");
}

pub fn make_new_path(path: &str) -> String {
    assert!(cfg!(windows));
    // Windows just uses PATH as the library search path, so we have to
    // maintain the current value while adding our own
    match env::var(lib_path_env_var()) {
        Ok(curr) => {
            format!("{}{}{}", path, path_div(), curr)
        }
        Err(..) => path.to_string()
    }
}

pub fn lib_path_env_var() -> &'static str { "PATH" }
fn path_div() -> &'static str { ";" }

pub fn logv(config: &Config, s: String) {
    debug!("{}", s);
    if config.verbose { println!("{}", s); }
}
