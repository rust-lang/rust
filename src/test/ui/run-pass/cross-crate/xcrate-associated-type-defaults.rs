// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:xcrate_associated_type_defaults.rs

extern crate xcrate_associated_type_defaults;
use xcrate_associated_type_defaults::Foo;

struct LocalDefault;
impl Foo<u32> for LocalDefault {}

struct LocalOverride;
impl Foo<u64> for LocalOverride {
    type Out = bool;
}

fn main() {
    assert_eq!(
        <() as Foo<u32>>::Out::default().to_string(),
        "0");
    assert_eq!(
        <() as Foo<u64>>::Out::default().to_string(),
        "false");

    assert_eq!(
        <LocalDefault as Foo<u32>>::Out::default().to_string(),
        "0");
    assert_eq!(
        <LocalOverride as Foo<u64>>::Out::default().to_string(),
        "false");
}
