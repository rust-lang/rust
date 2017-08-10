// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::env;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Command;

/// Get the name of the environment variable that holds dynamic library
/// locations
pub fn dylib_env_var() -> &'static str {
    if cfg!(windows) {
        "PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else if cfg!(target_os = "haiku") {
        "LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    }
}

/// Add `lib_path` and `aux_path` (if it is `Some`) to the dynamic library
/// env var
pub fn add_target_env(cmd: &mut Command, lib_path: &str, aux_path: Option<&str>) {
    // Need to be sure to put both the lib_path and the aux path in the dylib
    // search path for the child.
    let var = dylib_env_var();
    let mut path = env::split_paths(&env::var_os(var).unwrap_or(OsString::new()))
        .collect::<Vec<_>>();
    if let Some(p) = aux_path {
        path.insert(0, PathBuf::from(p))
    }
    path.insert(0, PathBuf::from(lib_path));

    // Add the new dylib search path var
    let newpath = env::join_paths(&path).unwrap();
    cmd.env(var, newpath);
}
