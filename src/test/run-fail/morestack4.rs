// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test newsched transition
// error-pattern:explicit failure

// Just testing unwinding

extern mod extra;

use std::task;

fn getbig_and_fail(i: int) {
    let r = and_then_get_big_again(5);
    if i != 0 {
        getbig_and_fail(i - 1);
    } else {
        fail2!();
    }
}

struct and_then_get_big_again {
  x:int,
}

impl Drop for and_then_get_big_again {
    fn drop(&mut self) {}
}

fn and_then_get_big_again(x:int) -> and_then_get_big_again {
    and_then_get_big_again {
        x: x
    }
}

fn main() {
    do task::spawn {
        getbig_and_fail(1);
    };
}
