// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn a<F:Fn(isize, isize) -> isize>(mut f: F) {
    let g = &mut f;
    f(1, 2);    //~ ERROR cannot borrow `f` as immutable
    //~^ ERROR cannot borrow `f` as immutable
}

fn b<F:FnMut(isize, isize) -> isize>(f: F) {
    f(1, 2);    //~ ERROR cannot borrow immutable argument
}

fn c<F:FnOnce(isize, isize) -> isize>(f: F) {
    f(1, 2);
    f(1, 2);    //~ ERROR use of moved value
}

fn main() {}
