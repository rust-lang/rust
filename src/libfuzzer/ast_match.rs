// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std;
use vec;

fn vec_equal<T>(v: ~[T],
                u: ~[T],
                element_equality_test: @fn(&&T, &&T) -> bool) ->
   bool {
    let Lv = vec::len(v);
    if Lv != vec::len(u) { return false; }
    let i = 0u;
    while i < Lv {
        if !element_equality_test(v[i], u[i]) { return false; }
        i += 1u;
    }
    return true;
}

fn builtin_equal<T>(&&a: T, &&b: T) -> bool { return a == b; }
fn builtin_equal_int(&&a: int, &&b: int) -> bool { return a == b; }

fn main() {
    assert!((builtin_equal(5, 5)));
    assert!((!builtin_equal(5, 4)));
    assert!((!vec_equal(~[5, 5], ~[5], bind builtin_equal(_, _))));
    assert!((!vec_equal(~[5, 5], ~[5], builtin_equal_int)));
    assert!((!vec_equal(~[5, 5], ~[5, 4], builtin_equal_int)));
    assert!((!vec_equal(~[5, 5], ~[4, 5], builtin_equal_int)));
    assert!((vec_equal(~[5, 5], ~[5, 5], builtin_equal_int)));

    error!("Pass");
}
