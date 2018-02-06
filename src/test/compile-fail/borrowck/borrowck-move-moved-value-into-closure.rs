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

#![feature(box_syntax)]

fn call_f<F:FnOnce() -> isize>(f: F) -> isize {
    f()
}

fn main() {
    let t: Box<_> = box 3;

    call_f(move|| { *t + 1 });
    call_f(move|| { *t + 1 }); //[ast]~ ERROR capture of moved value
    //[mir]~^ ERROR use of moved value
}
