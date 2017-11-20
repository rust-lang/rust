// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that unboxed closures cannot capture their own type.
//
// Also regression test for issue #21410.

fn g<F>(_: F) where F: FnOnce(Option<F>) {}

fn main() {
    g(|_| {  }); //~ ERROR closure/generator type that references itself
}
