// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::process::Command;
use std::env;

fn check_var(&(ref key, ref val): &(String, String)) -> bool {
    match &**key {
        "FOO" => { assert_eq!(val, "BAR"); true },
        "BAR" => { assert_eq!(val, "BAZ"); true },
        "BAZ" => { assert_eq!(val, "FOO"); true },
        _ => false,
    }
}

fn main() {
    if let Some(arg) = env::args().nth(1) {
        match &*arg {
            "empty" => {
                assert_eq!(env::vars().count(), 0);
                assert_eq!(env::vars().next(), None);
                assert_eq!(env::vars().nth(1), None);
                assert_eq!(env::vars().last(), None);
            },
            "many" => {
                assert!(env::vars().count() >= 3);
                assert!(env::vars().last().is_some());
                assert_eq!(env::vars().filter(check_var).count(), 3);
                assert_eq!(
                    (0..env::vars().count())
                        .map(|i| env::vars().nth(i).unwrap())
                        .filter(check_var)
                        .count(),
                    3);
            },
            arg => {
                panic!("Unexpected arg {}", arg);
            },
        }
    } else {
        // Command::env_clear does not work on Windows.
        // https://github.com/rust-lang/rust/issues/31259
        if !cfg!(windows) {
            let status = Command::new(env::current_exe().unwrap())
                                 .arg("empty")
                                 .env_clear()
                                 .status()
                                 .unwrap();
            assert_eq!(status.code(), Some(0));
        }

        let status = Command::new(env::current_exe().unwrap())
                             .arg("many")
                             .env("FOO", "BAR")
                             .env("BAR", "BAZ")
                             .env("BAZ", "FOO")
                             .status()
                             .unwrap();
        assert_eq!(status.code(), Some(0));
    }
}
