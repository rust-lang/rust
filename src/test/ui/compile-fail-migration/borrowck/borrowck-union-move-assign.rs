// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(untagged_unions)]

// Non-copy
struct A;
struct B;

union U {
    a: A,
    b: B,
}

fn main() {
    unsafe {
        {
            let mut u = U { a: A };
            let a = u.a;
            let a = u.a; //~ ERROR use of moved value: `u.a`
        }
        {
            let mut u = U { a: A };
            let a = u.a;
            u.a = A;
            let a = u.a; // OK
        }
        {
            let mut u = U { a: A };
            let a = u.a;
            u.b = B;
            let a = u.a; // OK
        }
    }
}
