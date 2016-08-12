// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test correct type inference for projections when the associated
// type has multiple trait bounds

struct A;
struct B;

trait Foo {
    type T: PartialEq<A> + PartialEq<B>;
}

fn generic<F: Foo>(t: F::T, a: A, b: B) -> bool {
    // Equivalent, but not as explicit: t == a && t == b
    <<F as Foo>::T as PartialEq<_>>::eq(&t, &a) &&
    <<F as Foo>::T as PartialEq<_>>::eq(&t, &b)
}

fn main() {}
