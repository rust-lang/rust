// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_patterns)]
#![feature(box_syntax)]

fn arg_item(box ref x: Box<isize>) -> &'static isize {
    x //~^ ERROR borrowed value does not live long enough
}

fn with<R, F>(f: F) -> R where F: FnOnce(Box<isize>) -> R { f(box 3) }

fn arg_closure() -> &'static isize {
    with(|box ref x| x) //~ ERROR borrowed value does not live long enough
}

fn main() {}
