// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten

#![feature(process_try_wait)]

use std::env;
use std::io;
use std::process::Command;
use std::thread;
use std::time::Duration;

fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 1 {
        match &args[1][..] {
            "sleep" => thread::sleep(Duration::new(1_000, 0)),
            _ => {}
        }
        return
    }

    let mut me = Command::new(env::current_exe().unwrap())
                         .arg("sleep")
                         .spawn()
                         .unwrap();
    let err = me.try_wait().unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::WouldBlock);
    let err = me.try_wait().unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::WouldBlock);

    me.kill().unwrap();
    me.wait().unwrap();

    let status = me.try_wait().unwrap();
    assert!(!status.success());
    let status = me.try_wait().unwrap();
    assert!(!status.success());

    let mut me = Command::new(env::current_exe().unwrap())
                         .arg("return-quickly")
                         .spawn()
                         .unwrap();
    loop {
        match me.try_wait() {
            Ok(res) => {
                assert!(res.success());
                break
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(1));
            }
            Err(e) => panic!("error in try_wait: {}", e),
        }
    }

    let status = me.try_wait().unwrap();
    assert!(status.success());
}
