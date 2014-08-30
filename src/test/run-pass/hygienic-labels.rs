// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(macro_rules)]

macro_rules! loop_x {
    ($e: expr) => {
        // $e shouldn't be able to interact with this 'x
        'x: loop { $e }
    }
}

macro_rules! run_once {
    ($e: expr) => {
        // ditto
        'x: for _ in range(0i, 1) { $e }
    }
}

macro_rules! while_x {
    ($e: expr) => {
        // ditto
        'x: while 1i + 1 == 2 { $e }
    }
}

pub fn main() {
    'x: for _ in range(0i, 1) {
        // this 'x should refer to the outer loop, lexically
        loop_x!(break 'x);
        fail!("break doesn't act hygienically inside for loop");
    }

    'x: loop {
        // ditto
        loop_x!(break 'x);
        fail!("break doesn't act hygienically inside infinite loop");
    }

    'x: while 1i + 1 == 2 {
        while_x!(break 'x);
        fail!("break doesn't act hygienically inside infinite while loop");
    }

    'x: for _ in range(0i, 1) {
        // ditto
        run_once!(continue 'x);
        fail!("continue doesn't act hygienically inside for loop");
    }
}
