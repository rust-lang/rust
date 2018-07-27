// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(existential_type)]
// compile-pass

trait Bar {}
struct Dummy<U>(U);
impl<V> Bar for Dummy<V> {}

trait Foo<T> {
    type Assoc: Bar;
    fn foo(t: T) -> Self::Assoc;
}

impl<W> Foo<W> for i32 {
    existential type Assoc: Bar;
    fn foo(w: W) -> Self::Assoc {
        Dummy(w)
    }
}

struct NonGeneric;
impl Bar for NonGeneric {}

impl<W> Foo<W> for u32 {
    existential type Assoc: Bar;
    fn foo(_: W) -> Self::Assoc {
        NonGeneric
    }
}

fn main() {}
