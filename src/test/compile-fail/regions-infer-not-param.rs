// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct direct {
    f: &int
}

struct indirect1 {
    g: fn@(direct)
}

struct indirect2 {
    g: fn@(direct/&)
}

struct indirect3 {
    g: fn@(direct/&self)
}

fn take_direct(p: direct) -> direct { p } //~ ERROR mismatched types
fn take_indirect1(p: indirect1) -> indirect1 { p }
fn take_indirect2(p: indirect2) -> indirect2 { p }
fn take_indirect3(p: indirect3) -> indirect3 { p } //~ ERROR mismatched types
fn main() {}
