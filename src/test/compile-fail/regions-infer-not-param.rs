// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct direct<'a> {
    f: &'a int
}

struct indirect1 {
    // Here the lifetime parameter of direct is bound by the fn()
    g: |direct|: 'static
}

struct indirect2<'a> {
    // But here it is set to 'a
    g: |direct<'a>|: 'static
}

fn take_direct<'a,'b>(p: direct<'a>) -> direct<'b> { p } //~ ERROR mismatched types

fn take_indirect1(p: indirect1) -> indirect1 { p }

fn take_indirect2<'a,'b>(p: indirect2<'a>) -> indirect2<'b> { p } //~ ERROR mismatched types

fn main() {}
