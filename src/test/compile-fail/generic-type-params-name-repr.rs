// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(default_type_params)];

struct A;
struct B;
struct C;
struct Foo<T = A, U = B, V = C>;

fn main() {
    // Ensure that the printed type doesn't include the default type params...
    let _: Foo<int> = ();
    //~^ ERROR mismatched types: expected `Foo<int>` but found `()`

    // ...even when they're present, but the same types as the defaults.
    let _: Foo<int, B, C> = ();
    //~^ ERROR mismatched types: expected `Foo<int>` but found `()`

    // But not when there's a different type in between.
    let _: Foo<A, int, C> = ();
    //~^ ERROR mismatched types: expected `Foo<A,int>` but found `()`

    // And don't print <> at all when there's just defaults.
    let _: Foo<A, B, C> = ();
    //~^ ERROR mismatched types: expected `Foo` but found `()`
}
