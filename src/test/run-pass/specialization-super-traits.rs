// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(specialization)]

// Test that you can specialize via an explicit trait hierarchy

// FIXME: this doesn't work yet...

trait Parent {}
trait Child: Parent {}

trait Foo {}

impl<T: Parent> Foo for T {}
impl<T: Child> Foo for T {}

fn main() {}
