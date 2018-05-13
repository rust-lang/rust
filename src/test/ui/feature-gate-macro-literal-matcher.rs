// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the :lifetime macro fragment cannot be used when macro_lifetime_matcher
// feature gate is not used.

macro_rules! m { ($lt:literal) => {} }
//~^ ERROR :literal fragment specifier is experimental and subject to change

fn main() {
    m!("some string literal");
}
