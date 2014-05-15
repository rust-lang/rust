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

#[cfg(target_os = "win32")]
fn target_env(lib_path: &str, prog: &str) -> Vec<(StrBuf, StrBuf)> {
    let env = os::env();

    // Make sure we include the aux directory in the path
    assert!(prog.ends_with(".exe"));
    let aux_path = prog.slice(0u, prog.len() - 4u).to_owned() + ".libaux";

    let mut new_env: Vec<_> = env.move_iter().map(|(k, v)| {
        let new_v = if "PATH" == k {
            format_strbuf!("{};{};{}", v, lib_path, aux_path)
        } else {
            v.to_strbuf()
        };
        (k.to_strbuf(), new_v)
    }).collect();
    if prog.ends_with("rustc.exe") {
        new_env.push(("RUST_THREADS".to_strbuf(), "1".to_strbuf()));
    }
    return new_env;
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
fn target_env(lib_path: &str, prog: &str) -> Vec<(StrBuf,StrBuf)> {
    // Make sure we include the aux directory in the path
    let aux_path = prog + ".libaux";

    let mut env: Vec<(StrBuf,StrBuf)> =
        os::env().move_iter()
                 .map(|(ref k, ref v)| (k.to_strbuf(), v.to_strbuf()))
                 .collect();
    let var = if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    };
    let prev = match env.iter().position(|&(ref k, _)| k.as_slice() == var) {
        Some(i) => env.remove(i).unwrap().val1(),
        None => "".to_strbuf(),
    };
    env.push((var.to_strbuf(), if prev.is_empty() {
        format_strbuf!("{}:{}", lib_path, aux_path)
    } else {
        format_strbuf!("{}:{}:{}", lib_path, aux_path, prev)
    }));
    return env;
}

pub struct Result {pub status: ProcessExit, pub out: StrBuf, pub err: StrBuf}

pub fn run(lib_path: &str,
           prog: &str,
           args: &[StrBuf],
           env: Vec<(StrBuf, StrBuf)> ,
           input: Option<StrBuf>) -> Option<Result> {

    let env = env.clone().append(target_env(lib_path, prog).as_slice());
    match Command::new(prog).args(args).env(env.as_slice()).spawn() {
        Ok(mut process) => {
            for input in input.iter() {
                process.stdin.get_mut_ref().write(input.as_bytes()).unwrap();
            }
            let ProcessOutput { status, output, error } =
                process.wait_with_output().unwrap();

            Some(Result {
                status: status,
                out: str::from_utf8(output.as_slice()).unwrap().to_strbuf(),
                err: str::from_utf8(error.as_slice()).unwrap().to_strbuf()
            })
        },
        Err(..) => None
    }
}

pub fn run_background(lib_path: &str,
           prog: &str,
           args: &[StrBuf],
           env: Vec<(StrBuf, StrBuf)> ,
           input: Option<StrBuf>) -> Option<Process> {

    let env = env.clone().append(target_env(lib_path, prog).as_slice());
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
