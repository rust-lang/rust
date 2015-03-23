// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #21726: an issue arose around the rules for
// subtyping of projection types that resulted in an unconstrained
// region, yielding region inference failures.

// pretty-expanded FIXME #23616

fn main() { }

fn foo<'a>(s: &'a str) {
    let b: B<()> = B::new(s, ());
    b.get_short();
}

trait IntoRef<'a> {
    type T: Clone;
    fn into_ref(self, &'a str) -> Self::T;
}

impl<'a> IntoRef<'a> for () {
    type T = &'a str;
    fn into_ref(self, s: &'a str) -> &'a str {
        s
    }
}

struct B<'a, P: IntoRef<'a>>(P::T);

impl<'a, P: IntoRef<'a>> B<'a, P> {
    fn new(s: &'a str, i: P) -> B<'a, P> {
        B(i.into_ref(s))
    }

    fn get_short(&self) -> P::T {
        self.0.clone()
    }
}
