// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Direct<'a> {
    f: &'a isize
}

struct Indirect1 {
    // Here the lifetime parameter of direct is bound by the fn()
    g: Box<FnOnce(Direct) + 'static>
}

struct Indirect2<'a> {
    // But here it is set to 'a
    g: Box<FnOnce(Direct<'a>) + 'static>
}

fn take_direct<'a,'b>(p: Direct<'a>) -> Direct<'b> { p } //~ ERROR mismatched types

fn take_indirect1(p: Indirect1) -> Indirect1 { p }

fn take_indirect2<'a,'b>(p: Indirect2<'a>) -> Indirect2<'b> { p } //~ ERROR mismatched types
//~| expected type `Indirect2<'b>`
//~| found type `Indirect2<'a>`
//~| ERROR mismatched types
//~| expected type `Indirect2<'b>`
//~| found type `Indirect2<'a>`

fn main() {}
