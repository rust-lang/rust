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
fn target_env(lib_path: &str, prog: &str) -> ~[(~str,~str)] {

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
fn target_env(_lib_path: &str, _prog: &str) -> ~[(~str,~str)] {
    os::env()
}

pub struct Result {status: ProcessExit, out: ~str, err: ~str}

pub fn run(lib_path: &str,
           prog: &str,
           args: &[~str],
           env: ~[(~str, ~str)],
           input: Option<~str>) -> Option<Result> {

    let env = env + target_env(lib_path, prog);
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
           env: ~[(~str, ~str)],
           input: Option<~str>) -> Option<Process> {

    let env = env + target_env(lib_path, prog);
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
