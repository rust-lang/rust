// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo: Fn(i32) -> i32 + Send {}

impl<T: ?Sized + Fn(i32) -> i32 + Send> Foo for T {}

fn wants_foo(f: Box<Foo>) -> i32 {
    f(42)
}

fn main() {
    let f = Box::new(|x| x);
    assert_eq!(wants_foo(f), 42);
}
