// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-windows - this is a unix-specific test
// ignore-pretty issue #37199
// ignore-emscripten
#![feature(process_exec)]

use std::env;
use std::os::unix::process::CommandExt;
use std::process::Command;

fn main() {
    let mut args = env::args();
    let me = args.next().unwrap();

    if let Some(arg) = args.next() {
        match &arg[..] {
            "test1" => println!("passed"),

            "exec-test1" => {
                let err = Command::new(&me).arg("test1").exec();
                panic!("failed to spawn: {}", err);
            }

            "exec-test2" => {
                Command::new("/path/to/nowhere").exec();
                println!("passed");
            }

            "exec-test3" => {
                Command::new(&me).arg("bad\0").exec();
                println!("passed");
            }

            "exec-test4" => {
                Command::new(&me).current_dir("/path/to/nowhere").exec();
                println!("passed");
            }

            _ => panic!("unknown argument: {}", arg),
        }
        return
    }

    let output = Command::new(&me).arg("exec-test1").output().unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout, b"passed\n");

    let output = Command::new(&me).arg("exec-test2").output().unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout, b"passed\n");

    let output = Command::new(&me).arg("exec-test3").output().unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout, b"passed\n");

    let output = Command::new(&me).arg("exec-test4").output().unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout, b"passed\n");
}
