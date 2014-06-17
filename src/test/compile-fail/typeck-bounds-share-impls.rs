// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// Verify that `Share` bounds are correctly enforced.

struct MyStruct;

struct NotShareStruct<'a> {
    a: *MyStruct, // *-pointers are share if T is share
    //b: &'a MyStruct, // &-pointers are share if T is share
    //c: &'a mut MyStruct, // &mut-pointers are share if T is share
    d: MyStruct
}

impl<'a> Share for NotShareStruct<'a> {}
//~^ ERROR cannot implement the trait `core::kinds::Share` on type `NotShareStruct<a>` because the field with type `MyStruct` doesn't fulfill such trait.

enum Foo {
    Var(MyStruct)
}

fn share<T:Share>(_: T) {}

fn main() {
    share(Var(MyStruct));
    // FIXME(flaper87): This still doesn't typeck
    //ERROR type parameter with an incompatible type
}
