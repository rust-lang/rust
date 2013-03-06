// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod std;

fn test_heap_to_heap() {
    // a spills onto the heap
    let mut a = ~[0, 1, 2, 3, 4];
    a = a + a; // FIXME(#3387)---can't write a += a
    fail_unless!((vec::len(a) == 10u));
    fail_unless!((a[0] == 0));
    fail_unless!((a[1] == 1));
    fail_unless!((a[2] == 2));
    fail_unless!((a[3] == 3));
    fail_unless!((a[4] == 4));
    fail_unless!((a[5] == 0));
    fail_unless!((a[6] == 1));
    fail_unless!((a[7] == 2));
    fail_unless!((a[8] == 3));
    fail_unless!((a[9] == 4));
}

fn test_stack_to_heap() {
    // a is entirely on the stack
    let mut a = ~[0, 1, 2];
    // a spills to the heap
    a = a + a; // FIXME(#3387)---can't write a += a
    fail_unless!((vec::len(a) == 6u));
    fail_unless!((a[0] == 0));
    fail_unless!((a[1] == 1));
    fail_unless!((a[2] == 2));
    fail_unless!((a[3] == 0));
    fail_unless!((a[4] == 1));
    fail_unless!((a[5] == 2));
}

fn test_loop() {
    // Make sure we properly handle repeated self-appends.
    let mut a: ~[int] = ~[0];
    let mut i = 20;
    let mut expected_len = 1u;
    while i > 0 {
        log(error, vec::len(a));
        fail_unless!((vec::len(a) == expected_len));
        a = a + a; // FIXME(#3387)---can't write a += a
        i -= 1;
        expected_len *= 2u;
    }
}

pub fn main() { test_heap_to_heap(); test_stack_to_heap(); test_loop(); }
