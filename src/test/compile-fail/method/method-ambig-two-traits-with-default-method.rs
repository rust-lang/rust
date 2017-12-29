// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we correctly report an ambiguity where two applicable traits
// are in scope and the method being invoked is a default method not
// defined directly in the impl.

trait Foo { fn method(&self) {} }
trait Bar { fn method(&self) {} }

impl Foo for usize {}
impl Bar for usize {}

fn main() {
    1_usize.method(); //~ ERROR E0034
}
