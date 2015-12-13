// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-macos this needs valgrind 3.11 or higher; see
// https://github.com/rust-lang/rust/pull/30365#issuecomment-165763679

use std::env;
use std::process::{exit, Command};

fn main() {
    if env::args().len() > 1 {
        print!("hello!");
        exit(0);
    } else {
        let out = Command::new(env::args().next().unwrap()).arg("foo")
                          .output().unwrap();
        assert!(out.status.success());
        assert_eq!(String::from_utf8(out.stdout).unwrap(), "hello!");
        assert_eq!(String::from_utf8(out.stderr).unwrap(), "");
    }
}
