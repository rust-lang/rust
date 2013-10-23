// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

// Test that an object type `@Foo` is not considered to implement the
// trait `Foo`. Issue #5087.

trait Foo {}
fn take_foo<F:Foo>(f: F) {}
fn take_object(f: @Foo) { take_foo(f); } //~ ERROR failed to find an implementation of trait
fn main() {}
