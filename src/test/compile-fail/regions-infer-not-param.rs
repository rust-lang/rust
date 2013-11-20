// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct direct<'self> {
    f: &'self int
}

struct indirect1 {
    // Here the lifetime parameter of direct is bound by the fn()
    g: 'static |direct|
}

struct indirect2<'self> {
    // But here it is set to 'self
    g: 'static |direct<'self>|
}

fn take_direct(p: direct) -> direct { p } //~ ERROR mismatched types
//~^ ERROR cannot infer an appropriate lifetime

fn take_indirect1(p: indirect1) -> indirect1 { p }

fn take_indirect2(p: indirect2) -> indirect2 { p } //~ ERROR mismatched types
//~^ ERROR cannot infer an appropriate lifetime

fn main() {}
