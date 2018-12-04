// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo: Iterator<Item = i32, Item = i32> {}
//~^ ERROR is already specified

type Unit = ();

fn test() -> Box<Iterator<Item = (), Item = Unit>> {
//~^ ERROR is already specified
    Box::new(None.into_iter())
}

fn main() {
    let _: &Iterator<Item = i32, Item = i32>;
    test();
}
