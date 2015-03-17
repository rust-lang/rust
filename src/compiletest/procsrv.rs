// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(deprecated)] // for old path, for dynamic_lib

use std::dynamic_lib::DynamicLibrary;
use std::io::prelude::*;
use std::old_path::Path;
use std::process::{ExitStatus, Command, Child, Output, Stdio};

fn add_target_env(cmd: &mut Command, lib_path: &str, aux_path: Option<&str>) {
    // Need to be sure to put both the lib_path and the aux path in the dylib
    // search path for the child.
    let mut path = DynamicLibrary::search_path();
    match aux_path {
        Some(p) => path.insert(0, Path::new(p)),
        None => {}
    }
    path.insert(0, Path::new(lib_path));

    // Add the new dylib search path var
    let var = DynamicLibrary::envvar();
    let newpath = DynamicLibrary::create_path(&path);
    let newpath = String::from_utf8(newpath).unwrap();
    cmd.env(var, &newpath);
}

pub struct Result {pub status: ExitStatus, pub out: String, pub err: String}

pub fn run(lib_path: &str,
           prog: &str,
           aux_path: Option<&str>,
           args: &[String],
           env: Vec<(String, String)> ,
           input: Option<String>) -> Option<Result> {

    let mut cmd = Command::new(prog);
    cmd.args(args)
       .stdin(Stdio::piped())
       .stdout(Stdio::piped())
       .stderr(Stdio::piped());
    add_target_env(&mut cmd, lib_path, aux_path);
    for (key, val) in env {
        cmd.env(&key, &val);
    }

    match cmd.spawn() {
        Ok(mut process) => {
            if let Some(input) = input {
                process.stdin.as_mut().unwrap().write_all(input.as_bytes()).unwrap();
            }
            let Output { status, stdout, stderr } =
                process.wait_with_output().unwrap();

            Some(Result {
                status: status,
                out: String::from_utf8(stdout).unwrap(),
                err: String::from_utf8(stderr).unwrap()
            })
        },
        Err(..) => None
    }
}

pub fn run_background(lib_path: &str,
           prog: &str,
           aux_path: Option<&str>,
           args: &[String],
           env: Vec<(String, String)> ,
           input: Option<String>) -> Option<Child> {

    let mut cmd = Command::new(prog);
    cmd.args(args)
       .stdin(Stdio::piped())
       .stdout(Stdio::piped())
       .stderr(Stdio::piped());
    add_target_env(&mut cmd, lib_path, aux_path);
    for (key, val) in env {
        cmd.env(&key, &val);
    }

    match cmd.spawn() {
        Ok(mut process) => {
            if let Some(input) = input {
                process.stdin.as_mut().unwrap().write_all(input.as_bytes()).unwrap();
            }

            Some(process)
        },
        Err(..) => None
    }
}
