// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests the new destructor semantics.

use std::cell::RefCell;

fn main() {
    let b = {
        let a = Box::new(RefCell::new(4));
        *a.borrow() + 1    //~ ERROR `*a` does not live long enough
    };
    println!("{}", b);
}
