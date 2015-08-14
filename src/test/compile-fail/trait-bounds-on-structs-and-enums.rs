// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Trait {}

struct Foo<T:Trait> {
    x: T,
}

enum Bar<T:Trait> {
    ABar(isize),
    BBar(T),
    CBar(usize),
}

impl<T> Foo<T> {
//~^ ERROR the trait `Trait` is not implemented
    fn uhoh() {}
}

struct Baz {
    a: Foo<isize>, //~ ERROR not implemented
}

enum Boo {
    Quux(Bar<usize>), //~ ERROR not implemented
}

struct Badness<U> {
    b: Foo<U>, //~ ERROR not implemented
}

enum MoreBadness<V> {
    EvenMoreBadness(Bar<V>), //~ ERROR not implemented
}

struct TupleLike(
    Foo<i32>, //~ ERROR not implemented
);

enum Enum {
    DictionaryLike { field: Bar<u8> }, //~ ERROR not implemented
}

fn main() {
}
