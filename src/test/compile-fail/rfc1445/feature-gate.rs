// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that structural match is only permitted with a feature gate,
// and that if a feature gate is supplied, it permits the type to be
// used in a match.

// revisions: with_gate no_gate

// gate-test-structural_match

#![allow(dead_code)]
#![deny(future_incompatible)]
#![feature(rustc_attrs)]
#![cfg_attr(with_gate, feature(structural_match))]

#[structural_match] //[no_gate]~ ERROR semantics of constant patterns is not yet settled
struct Foo {
    x: u32
}

const FOO: Foo = Foo { x: 0 };

#[rustc_error]
fn main() { //[with_gate]~ ERROR compilation successful
    let y = Foo { x: 1 };
    match y {
        FOO => { }
        _ => { }
    }
}
