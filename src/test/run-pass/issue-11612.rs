// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// #11612
// We weren't updating the auto adjustments with all the resolved
// type information after type check.

trait A { fn dummy(&self) { } }

struct B<'a, T:'a> {
    f: &'a T
}

impl<'a, T> A for B<'a, T> {}

fn foo(_: &A) {}

fn bar<G>(b: &B<G>) {
    foo(b);       // Coercion should work
    foo(b as &A); // Explicit cast should work as well
}

fn main() {}
