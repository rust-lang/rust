// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
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

use std::rc::Rc;

pub fn main() {
    let _x = Rc::new(vec![1, 2]).into_iter();
    //[ast]~^ ERROR cannot move out of borrowed content [E0507]
    //[mir]~^^ ERROR [E0507]
}
