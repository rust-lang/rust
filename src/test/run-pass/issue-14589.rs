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

// FIXME (#22405): Replace `Box::new` with `box` here when/if possible.

// pretty-expanded FIXME #23616

fn main() {
    send::<Box<Foo>>(Box::new(Output(0)));
    Test::<Box<Foo>>::foo(Box::new(Output(0)));
    Test::<Box<Foo>>::new().send(Box::new(Output(0)));
}

fn send<T>(_: T) {}

struct Test<T> { marker: std::marker::PhantomData<T> }
impl<T> Test<T> {
    fn new() -> Test<T> { Test { marker: ::std::marker::PhantomData } }
    fn foo(_: T) {}
    fn send(&self, _: T) {}
}

trait Foo { fn dummy(&self) { }}
struct Output(int);
impl Foo for Output {}
