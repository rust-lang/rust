// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(nll)]

struct Foo {
}

impl Foo {
    fn method(&mut self, foo: &mut Foo) {
    }
}

fn main() {
    let mut foo = Foo { };
    foo.method(&mut foo);
    //~^     cannot borrow `foo` as mutable more than once at a time
    //~^^    cannot borrow `foo` as mutable more than once at a time
}
