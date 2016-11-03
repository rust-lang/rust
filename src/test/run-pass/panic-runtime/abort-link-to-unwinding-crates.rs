// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-C panic=abort
// aux-build:exit-success-if-unwind.rs
// no-prefer-dynamic
// ignore-emscripten Function not implemented

extern crate exit_success_if_unwind;

use std::process::Command;
use std::env;

fn main() {
    let mut args = env::args_os();
    let me = args.next().unwrap();

    if let Some(s) = args.next() {
        if &*s == "foo" {
            exit_success_if_unwind::bar(do_panic);
        }
    }
    let s = Command::new(env::args_os().next().unwrap()).arg("foo").status();
    assert!(s.unwrap().code() != Some(0));
}

fn do_panic() {
    panic!("try to catch me");
}
