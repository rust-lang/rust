// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass
// regression test for issue #50825
// Make sure that the `impl` bound (): X<T = ()> is preferred over
// the (): X bound in the where clause.

trait X {
    type T;
}

trait Y<U>: X {
    fn foo(x: &Self::T);
}

impl X for () {
    type T = ();
}

impl<T> Y<Vec<T>> for () where (): Y<T> {
    fn foo(_x: &()) {}
}

fn main () {}
