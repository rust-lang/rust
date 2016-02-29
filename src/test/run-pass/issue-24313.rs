// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten

use std::thread;
use std::env;
use std::process::Command;

struct Handle(i32);

impl Drop for Handle {
    fn drop(&mut self) { panic!(); }
}

thread_local!(static HANDLE: Handle = Handle(0));

fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.len() == 1 {
        let out = Command::new(&args[0]).arg("test").output().unwrap();
        let stderr = std::str::from_utf8(&out.stderr).unwrap();
        assert!(stderr.contains("panicked at 'explicit panic'"),
                "bad failure message:\n{}\n", stderr);
    } else {
        // TLS dtors are not always run on process exit
        thread::spawn(|| {
            HANDLE.with(|h| {
                println!("{}", h.0);
            });
        }).join().unwrap();
    }
}

