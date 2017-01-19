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
    ("android", "android"),
    ("bitrig", "bitrig"),
    ("darwin", "macos"),
    ("dragonfly", "dragonfly"),
    ("freebsd", "freebsd"),
    ("haiku", "haiku"),
    ("ios", "ios"),
    ("linux", "linux"),
    ("mingw32", "windows"),
    ("netbsd", "netbsd"),
    ("openbsd", "openbsd"),
    ("win32", "windows"),
    ("windows", "windows"),
    ("solaris", "solaris"),
    ("emscripten", "emscripten"),
];

const ARCH_TABLE: &'static [(&'static str, &'static str)] = &[
    ("aarch64", "aarch64"),
    ("amd64", "x86_64"),
    ("arm", "arm"),
    ("arm64", "aarch64"),
    ("hexagon", "hexagon"),
    ("i386", "x86"),
    ("i586", "x86"),
    ("i686", "x86"),
    ("mips", "mips"),
    ("msp430", "msp430"),
    ("powerpc", "powerpc"),
    ("powerpc64", "powerpc64"),
    ("s390x", "s390x"),
    ("sparc", "sparc"),
    ("x86_64", "x86_64"),
    ("xcore", "xcore"),
    ("asmjs", "asmjs"),
    ("wasm32", "wasm32"),
];

pub fn get_os(triple: &str) -> &'static str {
    for &(triple_os, os) in OS_TABLE {
        if triple.contains(triple_os) {
            return os;
        }
    }
    panic!("Cannot determine OS from triple");
}
pub fn get_arch(triple: &str) -> &'static str {
    for &(triple_arch, arch) in ARCH_TABLE {
        if triple.contains(triple_arch) {
            return arch;
        }
    }
    panic!("Cannot determine Architecture from triple");
}

pub fn get_env(triple: &str) -> Option<&str> {
    triple.split('-').nth(3)
}

pub fn make_new_path(path: &str) -> String {
    assert!(cfg!(windows));
    // Windows just uses PATH as the library search path, so we have to
    // maintain the current value while adding our own
    match env::var(lib_path_env_var()) {
        Ok(curr) => format!("{}{}{}", path, path_div(), curr),
        Err(..) => path.to_owned(),
    }
}

pub fn lib_path_env_var() -> &'static str {
    "PATH"
}
fn path_div() -> &'static str {
    ";"
}

pub fn logv(config: &Config, s: String) {
    debug!("{}", s);
    if config.verbose {
        println!("{}", s);
    }
}
