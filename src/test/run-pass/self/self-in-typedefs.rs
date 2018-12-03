// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass

#![feature(untagged_unions)]

#![allow(dead_code)]
#![allow(unions_with_drop_fields)]

enum A<'a, T: 'a>
where
    Self: Send, T: PartialEq<Self>
{
    Foo(&'a Self),
    Bar(T),
}

struct B<'a, T: 'a>
where
    Self: Send, T: PartialEq<Self>
{
    foo: &'a Self,
    bar: T,
}

union C<'a, T: 'a>
where
    Self: Send, T: PartialEq<Self>
{
    foo: &'a Self,
    bar: T,
}

fn main() {}
