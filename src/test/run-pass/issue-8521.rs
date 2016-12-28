// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo1 {}

trait A {}

macro_rules! foo1(($t:path) => {
    impl<T: $t> Foo1 for T {}
});

foo1!(A);

trait Foo2 {}

trait B<T> {}

#[allow(unused)]
struct C {}

macro_rules! foo2(($t:path) => {
    impl<T: $t> Foo2 for T {}
});

foo2!(B<C>);

fn main() {}
