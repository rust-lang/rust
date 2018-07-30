// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that while a trait with by-value self is object-safe, we
// can't actually invoke it from an object (yet...?).

#![feature(rustc_attrs)]

trait Bar {
    fn bar(self);
}

trait Baz {
    fn baz(self: Self);
}

fn use_bar(t: Box<Bar>) {
    t.bar() //~ ERROR cannot move a value of type (dyn Bar + 'static)
}

fn main() { }
