// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test which object types are considered sendable. This test
// is broken into two parts because some errors occur in distinct
// phases in the compiler. See kindck-send-object2.rs as well!

use std::marker::MarkerTrait;

fn assert_send<T:Send+'static>() { }
trait Dummy : MarkerTrait { }

// careful with object types, who knows what they close over...
fn test51<'a>() {
    assert_send::<&'a Dummy>();
    //~^ ERROR the trait `core::marker::Sync` is not implemented
}
fn test52<'a>() {
    assert_send::<&'a (Dummy+Sync)>();
    //~^ ERROR does not fulfill the required lifetime
}

// ...unless they are properly bounded
fn test60() {
    assert_send::<&'static (Dummy+Sync)>();
}
fn test61() {
    assert_send::<Box<Dummy+Send>>();
}

// closure and object types can have lifetime bounds which make
// them not ok
fn test_71<'a>() {
    assert_send::<Box<Dummy+'a>>();
    //~^ ERROR the trait `core::marker::Send` is not implemented
}

fn main() { }
