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
fn even(x: uint) -> bool {
    if x < 2u {
        return false;
    } else if x == 2u { return true; } else { return even(x - 2u); }
}

fn foo(x: uint) {
    if even(x) {
        info!("{}", x);
    } else {
        fail!("Number is odd");
    }
}

fn main() { foo(3u); }
