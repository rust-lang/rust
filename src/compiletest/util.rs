// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use common::config;

use std::io;
use std::os::getenv;

/// Conversion table from triple OS name to Rust SYSNAME
static OS_TABLE: &'static [(&'static str, &'static str)] = &[
    ("mingw32", "win32"),
    ("win32", "win32"),
    ("darwin", "macos"),
    ("android", "android"),
    ("linux", "linux"),
    ("freebsd", "freebsd"),
];

pub fn get_os(triple: &str) -> &'static str {
    for &(triple_os, os) in OS_TABLE.iter() {
        if triple.contains(triple_os) {
            return os
        }
    }
    fail!("Cannot determine OS from triple");
}

pub fn make_new_path(path: &str) -> ~str {

    // Windows just uses PATH as the library search path, so we have to
    // maintain the current value while adding our own
    match getenv(lib_path_env_var()) {
      Some(curr) => {
        format!("{}{}{}", path, path_div(), curr)
      }
      None => path.to_str()
    }
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "freebsd")]
pub fn lib_path_env_var() -> ~str { ~"LD_LIBRARY_PATH" }

#[cfg(target_os = "macos")]
pub fn lib_path_env_var() -> ~str { ~"DYLD_LIBRARY_PATH" }

#[cfg(target_os = "win32")]
pub fn lib_path_env_var() -> ~str { ~"PATH" }

#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
pub fn path_div() -> ~str { ~":" }

#[cfg(target_os = "win32")]
pub fn path_div() -> ~str { ~";" }

pub fn logv(config: &config, s: ~str) {
    debug!("{}", s);
    if config.verbose { io::println(s); }
}
