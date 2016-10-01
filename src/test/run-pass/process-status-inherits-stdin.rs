// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
// ignore-emscripten Function not implemented.

use std::env;
use std::io;
use std::io::Write;
use std::process::{Command, Stdio};

fn main() {
    let mut args = env::args();
    let me = args.next().unwrap();
    let arg = args.next();
    match arg.as_ref().map(|s| &s[..]) {
        None => {
            let mut s = Command::new(&me)
                                .arg("a1")
                                .stdin(Stdio::piped())
                                .spawn()
                                .unwrap();
            s.stdin.take().unwrap().write_all(b"foo\n").unwrap();
            let s = s.wait().unwrap();
            assert!(s.success());
        }
        Some("a1") => {
            let s = Command::new(&me).arg("a2").status().unwrap();
            assert!(s.success());
        }
        Some(..) => {
            let mut s = String::new();
            io::stdin().read_line(&mut s).unwrap();
            assert_eq!(s, "foo\n");
        }
    }
}
