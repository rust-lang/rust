// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we mutate a counter on the stack only when we expect to.

fn call<F>(f: F) where F : FnOnce() {
    f();
}

fn main() {
    let y = vec![format!("Hello"), format!("World")];
    let mut counter = 22_u32;

    call(|| {
        // Move `y`, but do not move `counter`, even though it is read
        // by value (note that it is also mutated).
        for item in y {
            let v = counter;
            counter += v;
        }
    });
    assert_eq!(counter, 88);

    call(move || {
        // this mutates a moved copy, and hence doesn't affect original
        counter += 1;
    });
    assert_eq!(counter, 88);
}
