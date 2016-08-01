// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that you can supply `&F` where `F: FnMut()`.

#![feature(lang_items)]

fn a<F:FnMut() -> i32>(mut f: F) -> i32 {
    f()
}

fn b(f: &mut FnMut() -> i32) -> i32 {
    a(f)
}

fn c<F:FnMut() -> i32>(f: &mut F) -> i32 {
    a(f)
}

fn main() {
    let z: isize = 7;

    let x = b(&mut || 22);
    assert_eq!(x, 22);

    let x = c(&mut || 22);
    assert_eq!(x, 22);
}
