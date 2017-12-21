// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(fn_traits)]

// That a closure whose expected argument types include two distinct
// bound regions.

use std::cell::Cell;

fn doit<T,F>(val: T, f: &F)
    where F : Fn(&Cell<&T>, &T)
{
    let x = Cell::new(&val);
    f.call((&x,&val))
}

pub fn main() {
    doit(0, &|x, y| {
        x.set(y); //~ ERROR E0312
    });
}
