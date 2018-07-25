// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![feature(const_fn)]

const fn max() -> usize { !0 }

trait Foo {
    const BAR: usize;
}

impl Foo for u32 {
    const BAR: usize = 42;
}

fn main() {
    let x: &'static usize = &max();
    let y: &'static usize = &std::usize::MAX;
    let z: &'static usize = &u32::BAR;
}
