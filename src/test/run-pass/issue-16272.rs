// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-aarch64
// ignore-emscripten

use std::process::Command;
use std::env;

fn main() {
    let len = env::args().len();

    if len == 1 {
        test();
    } else {
        assert_eq!(len, 3);
    }
}

fn test() {
    let status = Command::new(&env::current_exe().unwrap())
                         .arg("foo").arg("")
                         .status().unwrap();
    assert!(status.success());
}
