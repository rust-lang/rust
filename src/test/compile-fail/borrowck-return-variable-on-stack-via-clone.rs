// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that when we clone a `&T` pointer we properly relate the
// lifetime of the pointer which results to the pointer being cloned.
// Bugs in method resolution have sometimes broken this connection.
// Issue #19261.

fn leak<'a, T>(x: T) -> &'a T {
    (&x).clone() //~ ERROR `x` does not live long enough
}

fn main() { }
