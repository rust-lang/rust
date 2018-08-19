// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the NLL `relate_tys` code correctly deduces that a
// function returning either argument CANNOT be upcast to one
// that returns always its first argument.
//
// compile-flags:-Zno-leak-check

#![feature(nll)]

fn make_it() -> for<'a> fn(&'a u32, &'a u32) -> &'a u32 {
    panic!()
}

fn main() {
    let a: for<'a, 'b> fn(&'a u32, &'b u32) -> &'a u32 = make_it();
    //~^ ERROR higher-ranked subtype error
    drop(a);
}
