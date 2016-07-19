// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:Number is odd
fn even(x: usize) -> bool {
    if x < 2 {
        return false;
    } else if x == 2 {
        return true;
    } else {
        return even(x - 2);
    }
}

fn foo(x: usize) {
    if even(x) {
        println!("{}", x);
    } else {
        panic!("Number is odd");
    }
}

fn main() {
    foo(3);
}
