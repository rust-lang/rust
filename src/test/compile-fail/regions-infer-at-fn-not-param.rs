// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct parameterized1<'self> {
    g: &'self fn()
}

struct not_parameterized1 {
    g: &'static fn()
}

struct not_parameterized2 {
    g: &'static fn()
}

fn take1(p: parameterized1) -> parameterized1 { p }
//~^ ERROR mismatched types
//~^^ ERROR cannot infer an appropriate lifetime

fn take3(p: not_parameterized1) -> not_parameterized1 { p }
fn take4(p: not_parameterized2) -> not_parameterized2 { p }

fn main() {}
