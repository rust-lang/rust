// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::os;
use std::str;
use std::io::process::{ProcessExit, Command, Process, ProcessOutput};
use std::dynamic_lib::DynamicLibrary;

fn target_env(lib_path: &str, aux_path: Option<&str>) -> Vec<(String, String)> {
    // Need to be sure to put both the lib_path and the aux path in the dylib
    // search path for the child.
    let mut path = DynamicLibrary::search_path();
    match aux_path {
        Some(p) => path.insert(0, Path::new(p)),
        None => {}
    }
    path.insert(0, Path::new(lib_path));

    // Remove the previous dylib search path var
    let var = DynamicLibrary::envvar();
    let mut env: Vec<(String,String)> = os::env();
    match env.iter().position(|&(ref k, _)| k.as_slice() == var) {
        Some(i) => { env.remove(i); }
        None => {}
    }

    // Add the new dylib search path var
    let newpath = DynamicLibrary::create_path(path.as_slice());
    let newpath = str::from_utf8(newpath.as_slice()).unwrap().to_string();
    env.push((var.to_string(), newpath));
    return env;
}

pub struct Result {pub status: ProcessExit, pub out: String, pub err: String}

pub fn run(lib_path: &str,
           prog: &str,
           aux_path: Option<&str>,
           args: &[String],
           env: Vec<(String, String)> ,
           input: Option<String>) -> Option<Result> {

    let env = env.clone().append(target_env(lib_path, aux_path).as_slice());
    match Command::new(prog).args(args).env(env.as_slice()).spawn() {
        Ok(mut process) => {
            for input in input.iter() {
                process.stdin.get_mut_ref().write(input.as_bytes()).unwrap();
            }
            let ProcessOutput { status, output, error } =
                process.wait_with_output().unwrap();

            Some(Result {
                status: status,
                out: str::from_utf8(output.as_slice()).unwrap().to_string(),
                err: str::from_utf8(error.as_slice()).unwrap().to_string()
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
           input: Option<String>) -> Option<Process> {

    let env = env.clone().append(target_env(lib_path, aux_path).as_slice());
    match Command::new(prog).args(args).env(env.as_slice()).spawn() {
        Ok(mut process) => {
            for input in input.iter() {
                process.stdin.get_mut_ref().write(input.as_bytes()).unwrap();
            }

            Some(process)
        },
        Err(..) => None
    }
}
