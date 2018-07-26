// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo {}

trait Bar<T> {}

trait Iterable {
    type Item;
}

struct Container<T: Iterable<Item = impl Foo>> {
    //~^ ERROR `impl Trait` not allowed
    field: T
}

enum Enum<T: Iterable<Item = impl Foo>> {
    //~^ ERROR `impl Trait` not allowed
    A(T),
}

union Union<T: Iterable<Item = impl Foo> + Copy> {
    //~^ ERROR `impl Trait` not allowed
    x: T,
}

type Type<T: Iterable<Item = impl Foo>> = T;
//~^ ERROR `impl Trait` not allowed

fn main() {
}
