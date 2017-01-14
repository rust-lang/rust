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
use std::io::prelude::*;
use std::io;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Output, Stdio};

pub fn dylib_env_var() -> &'static str {
    if cfg!(windows) {
        "PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    }
}

fn add_target_env(cmd: &mut Command, lib_path: &str, aux_path: Option<&str>) {
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

pub struct Result {
    pub status: ExitStatus,
    pub out: String,
    pub err: String,
}

pub fn run(lib_path: &str,
           prog: &str,
           aux_path: Option<&str>,
           args: &[String],
           env: Vec<(String, String)>,
           input: Option<String>)
           -> io::Result<Result> {

    let mut cmd = Command::new(prog);
    cmd.args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    add_target_env(&mut cmd, lib_path, aux_path);
    for (key, val) in env {
        cmd.env(&key, &val);
    }

    let mut process = cmd.spawn()?;
    if let Some(input) = input {
        process.stdin.as_mut().unwrap().write_all(input.as_bytes()).unwrap();
    }
    let Output { status, stdout, stderr } = process.wait_with_output().unwrap();

    Ok(Result {
        status: status,
        out: String::from_utf8(stdout).unwrap(),
        err: String::from_utf8(stderr).unwrap(),
    })
}

pub fn run_background(lib_path: &str,
                      prog: &str,
                      aux_path: Option<&str>,
                      args: &[String],
                      env: Vec<(String, String)>,
                      input: Option<String>)
                      -> io::Result<Child> {

    let mut cmd = Command::new(prog);
    cmd.args(args)
       .stdin(Stdio::piped())
       .stdout(Stdio::piped());
    add_target_env(&mut cmd, lib_path, aux_path);
    for (key, val) in env {
        cmd.env(&key, &val);
    }

    let mut process = cmd.spawn()?;
    if let Some(input) = input {
        process.stdin.as_mut().unwrap().write_all(input.as_bytes()).unwrap();
    }

    Ok(process)
}
