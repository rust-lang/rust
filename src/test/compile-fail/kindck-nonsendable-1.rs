// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::gc::{Gc, GC};

fn foo(_x: Gc<uint>) {}

fn main() {
    let x = box(GC) 3u;
    let _: proc():Send = proc() foo(x); //~ ERROR `core::kinds::Send` is not implemented
    let _: proc():Send = proc() foo(x); //~ ERROR `core::kinds::Send` is not implemented
    let _: proc():Send = proc() foo(x); //~ ERROR `core::kinds::Send` is not implemented
    let _: proc() = proc() foo(x);
}
