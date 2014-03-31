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
use std::io::process::{ProcessExit, Process, ProcessConfig, ProcessOutput};

#[cfg(target_os = "win32")]
fn target_env(lib_path: &str, prog: &str) -> Vec<(~str,~str)> {

    let mut env = os::env();

    // Make sure we include the aux directory in the path
    assert!(prog.ends_with(".exe"));
    let aux_path = prog.slice(0u, prog.len() - 4u).to_owned() + ".libaux";

    env = env.map(|pair| {
        let (k,v) = (*pair).clone();
        if k == ~"PATH" { (~"PATH", v + ";" + lib_path + ";" + aux_path) }
        else { (k,v) }
    });
    if prog.ends_with("rustc.exe") {
        env.push((~"RUST_THREADS", ~"1"));
    }
    return env;
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
fn target_env(lib_path: &str, prog: &str) -> Vec<(~str,~str)> {
    // Make sure we include the aux directory in the path
    let aux_path = prog + ".libaux";

    let mut env: Vec<(~str,~str)> = os::env().move_iter().collect();
    let var = if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    };
    let prev = match env.iter().position(|&(ref k, _)| k.as_slice() == var) {
        Some(i) => env.remove(i).unwrap().val1(),
        None => ~"",
    };
    env.push((var.to_owned(), if prev.is_empty() {
        lib_path + ":" + aux_path
    } else {
        lib_path + ":" + aux_path + ":" + prev
    }));
    return env;
}

pub struct Result {pub status: ProcessExit, pub out: ~str, pub err: ~str}

pub fn run(lib_path: &str,
           prog: &str,
           args: &[~str],
           env: Vec<(~str, ~str)> ,
           input: Option<~str>) -> Option<Result> {

    let env = env.clone().append(target_env(lib_path, prog).as_slice());
    let mut opt_process = Process::configure(ProcessConfig {
        program: prog,
        args: args,
        env: Some(env.as_slice()),
        .. ProcessConfig::new()
    });

    match opt_process {
        Ok(ref mut process) => {
            for input in input.iter() {
                process.stdin.get_mut_ref().write(input.as_bytes()).unwrap();
            }
            let ProcessOutput { status, output, error } = process.wait_with_output();

            Some(Result {
                status: status,
                out: str::from_utf8_owned(output).unwrap(),
                err: str::from_utf8_owned(error).unwrap()
            })
        },
        Err(..) => None
    }
}

pub fn run_background(lib_path: &str,
           prog: &str,
           args: &[~str],
           env: Vec<(~str, ~str)> ,
           input: Option<~str>) -> Option<Process> {

    let env = env.clone().append(target_env(lib_path, prog).as_slice());
    let opt_process = Process::configure(ProcessConfig {
        program: prog,
        args: args,
        env: Some(env.as_slice()),
        .. ProcessConfig::new()
    });

    match opt_process {
        Ok(mut process) => {
            for input in input.iter() {
                process.stdin.get_mut_ref().write(input.as_bytes()).unwrap();
            }

            Some(process)
        },
        Err(..) => None
    }
}
