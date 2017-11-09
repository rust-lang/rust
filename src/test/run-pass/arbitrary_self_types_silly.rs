// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(arbitrary_self_types)]

struct Foo;
struct Bar;

impl std::ops::Deref for Bar {
    type Target = Foo;

    fn deref(&self) -> &Foo {
        &Foo
    }
}

impl Foo {
    fn bar(self: Bar) -> i32 { 3 }
}

fn main() {
    assert_eq!(3, Bar.bar());
}
