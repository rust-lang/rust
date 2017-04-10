// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is a fake compile fail test as there's no way to generate a
// `#![feature(linker_flavor)]` error. The only reason we have a `linker_flavor`
// feature gate is to be able to document `-Z linker-flavor` in the unstable
// book

#[used]
fn foo() {}
//~^^ ERROR the `#[used]` attribute is an experimental feature

fn main() {}
