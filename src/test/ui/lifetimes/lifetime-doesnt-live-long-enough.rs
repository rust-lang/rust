// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait ListItem<'a> {
    fn list_name() -> &'a str;
}

trait Collection { fn len(&self) -> usize; }

struct List<'a, T: ListItem<'a>> {
    slice: &'a [T]
    //~^ ERROR may not live long enough
}

impl<'a, T: ListItem<'a>> Collection for List<'a, T> {
    fn len(&self) -> usize {
        0
    }
}

struct Foo<T> {
    foo: &'static T
    //~^ ERROR may not live long enough
}

trait X<K>: Sized {
    fn foo<'a, L: X<&'a Nested<K>>>();
    //~^ ERROR may not live long enough

    // check that we give a sane error for `Self`
    fn bar<'a, L: X<&'a Nested<Self>>>();
    //~^ ERROR may not live long enough

    // check that we give a sane error for nested generics
    fn baz<'a, L, M: X<&'a Nested<L>>>() {
        //~^ ERROR may not live long enough
    }
}

trait TraitB {}

struct Nested<K>(K);
impl<K> Nested<K> {
    fn generic_in_parent<'a, L: X<&'a Nested<K>>>() {
        //~^ ERROR may not live long enough
    }
    fn generic_in_child<'a, 'b, L: X<&'a Nested<M>>, M: 'b>() {
        //~^ ERROR may not live long enough
    }
}

fn main() {}
