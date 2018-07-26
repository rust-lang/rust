// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![feature(existential_type)]

fn main() {}

// don't reveal the concrete type
existential type NoReveal: std::fmt::Debug;

fn define_no_reveal() -> NoReveal {
    ""
}

fn no_reveal(x: NoReveal) {
    let _: &'static str = x; //~ mismatched types
    let _ = x as &'static str; //~ non-primitive cast
}
