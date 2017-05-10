// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::env;
use std::io;
use std::io::{Read, Write};
use std::process::{Command, Stdio};

fn main(){
    if env::args().count() > 1 && env::args().nth(1) == Some("child".to_string()) {
        child()
    } else {
        let mut p = Command::new(env::args().nth(0).unwrap())
                .arg("child")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn().unwrap();
        {
            let mut buf = [0; 1];
            assert!(p.stdout.as_mut().unwrap().read(&mut buf).unwrap() >= 1);
            assert_eq!(buf[0], b'>');
            assert!(p.stdin.as_mut().unwrap().write(b"abcd\n").unwrap() >= 1);
        }
        // FIXME(#25572): timeout and fail on timeout
        assert!(p.wait().unwrap().success());
    }
}

fn child(){
    let stdout = io::stdout();
    let lstdout = stdout.lock();
    let mut stdin = io::stdin();
    print!(">");
    let mut letter = [0; 1];
    stdin.read(&mut letter).unwrap();
}
