// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:derive-b.rs

#[macro_use]
extern crate derive_b;

#[B] //~ ERROR `B` is ambiguous
#[C] //~ ERROR attribute `C` is currently unknown to the compiler
#[B(D)] //~ ERROR `B` is ambiguous
#[B(E = "foo")] //~ ERROR `B` is ambiguous
#[B(arbitrary tokens)] //~ ERROR `B` is ambiguous
                       //~^ ERROR expected one of `(`, `)`, `,`, `::`, or `=`, found `tokens`
#[derive(B)]
struct B;

fn main() {}
