// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
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

// Variation on `borrowck-use-uninitialized-in-cast` in which we do a
// trait cast from an uninitialized source. Issue #20791.

trait Foo { fn dummy(&self) { } }
impl Foo for i32 { }

fn main() {
    let x: &i32;
    let y = x as *const Foo; //[ast]~ ERROR use of possibly uninitialized variable: `*x`
                             //[mir]~^ ERROR [E0381]
}
