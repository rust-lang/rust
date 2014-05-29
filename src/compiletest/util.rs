// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use common::Config;

#[cfg(target_os = "win32")]
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

#[cfg(target_os = "win32")]
pub fn make_new_path(path: &str) -> String {

    // Windows just uses PATH as the library search path, so we have to
    // maintain the current value while adding our own
    match getenv(lib_path_env_var().as_slice()) {
      Some(curr) => {
        format!("{}{}{}", path, path_div(), curr)
      }
      None => path.to_str().to_string()
    }
}

#[cfg(target_os = "win32")]
pub fn lib_path_env_var() -> String { "PATH".to_string() }

#[cfg(target_os = "win32")]
pub fn path_div() -> String { ";".to_string() }

pub fn logv(config: &Config, s: String) {
    debug!("{}", s);
    if config.verbose { println!("{}", s); }
}
