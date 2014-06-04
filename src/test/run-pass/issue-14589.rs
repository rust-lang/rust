// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// All 3 expressions should work in that the argument gets
// coerced to a trait object

fn main() {
    send::<Box<Foo>>(box Output(0));
    Test::<Box<Foo>>::foo(box Output(0));
    Test::<Box<Foo>>.send(box Output(0));
}

fn send<T>(_: T) {}

struct Test<T>;
impl<T> Test<T> {
    fn foo(_: T) {}
    fn send(&self, _: T) {}
}

trait Foo {}
struct Output(int);
impl Foo for Output {}
