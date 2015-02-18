// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test an issue where we reported ambiguity between the where-clause
// and the blanket impl. The only important thing is that compilation
// succeeds here. Issue #22110.

#![allow(dead_code)]

trait Foo<A> {
    fn foo(&self, a: A);
}

impl<A,F:Fn(A)> Foo<A> for F {
    fn foo(&self, _: A) { }
}

fn baz<A,F:for<'a> Foo<(&'a A,)>>(_: F) { }

fn components<T,A>(t: fn(&A))
    where fn(&A) : for<'a> Foo<(&'a A,)>,
{
    baz(t)
}

fn main() {
}
