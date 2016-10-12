// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that labels injected by macros do not break hygiene.  This
// checks cases where the macros invocations are under the rhs of a
// let statement.

// Issue #24278: The label/lifetime shadowing checker from #24162
// conservatively ignores hygiene, and thus issues warnings that are
// both true- and false-positives for this test.

macro_rules! loop_x {
    ($e: expr) => {
        // $e shouldn't be able to interact with this 'x
        'x: loop { $e }
    }
}

macro_rules! while_true {
    ($e: expr) => {
        // $e shouldn't be able to interact with this 'x
        'x: while 1 + 1 == 2 { $e }
    }
}

macro_rules! run_once {
    ($e: expr) => {
        // ditto
        'x: for _ in 0..1 { $e }
    }
}

pub fn main() {
    let mut i = 0;

    let j: isize = {
        'x: loop {
            // this 'x should refer to the outer loop, lexically
            loop_x!(break 'x);
            i += 1;
        }
        i + 1
    };
    assert_eq!(j, 1);

    let k: isize = {
        'x: for _ in 0..1 {
            // ditto
            loop_x!(break 'x);
            i += 1;
        }
        i + 1
    };
    assert_eq!(k, 1);

    let l: isize = {
        'x: for _ in 0..1 {
            // ditto
            while_true!(break 'x);
            i += 1;
        }
        i + 1
    };
    assert_eq!(l, 1);

    let n: isize = {
        'x: for _ in 0..1 {
            // ditto
            run_once!(continue 'x);
            i += 1;
        }
        i + 1
    };
    assert_eq!(n, 1);
}
