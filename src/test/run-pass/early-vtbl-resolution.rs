// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


trait thing<A> {
    fn foo() -> Option<A>;
}
impl<A> int: thing<A> {
    fn foo() -> Option<A> { None }
}
fn foo_func<A, B: thing<A>>(x: B) -> Option<A> { x.foo() }

fn main() {

    for iter::eachi(&(Some({a: 0}))) |i, a| { 
        debug!("%u %d", i, a.a);
    }

    let _x: Option<float> = foo_func(0);
}
