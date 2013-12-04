// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::task;

pub fn main() { test05(); }

fn test05_start(f: proc(int)) {
    f(22);
}

fn test05() {
    let three = ~3;
    let fn_to_send: proc(int) = proc(n) {
        error!("{}", *three + n); // will copy x into the closure
        assert_eq!(*three, 3);
    };
    task::spawn(proc() {
        test05_start(fn_to_send);
    });
}
