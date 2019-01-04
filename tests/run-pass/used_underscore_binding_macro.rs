// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(clippy::useless_attribute)] //issue #2910
#![allow(clippy::random_state)] // issue #3628

#[macro_use]
extern crate serde_derive;

/// Test that we do not lint for unused underscores in a `MacroAttribute`
/// expansion
#[deny(clippy::used_underscore_binding)]
#[derive(Deserialize)]
struct MacroAttributesTest {
    _foo: u32,
}

#[test]
fn macro_attributes_test() {
    let _ = MacroAttributesTest { _foo: 0 };
}

fn main() {}
