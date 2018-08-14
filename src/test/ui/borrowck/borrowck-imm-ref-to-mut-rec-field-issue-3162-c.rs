// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

fn main() {
    let mut _a = 3;
    let b = &mut _a;
    {
        let c = &*b;
        _a = 4; //[ast]~ ERROR cannot assign to `_a`
        //[mir]~^ ERROR cannot assign to `_a` because it is borrowed
        drop(c);
    }
    drop(b);
}
