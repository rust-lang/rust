// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test a "pass-through" object-lifetime-default that produces errors.

#![allow(dead_code)]

trait SomeTrait {
    fn dummy(&self) { }
}

struct MyBox<T:?Sized> {
    r: Box<T>
}

fn deref<T>(ss: &T) -> T {
    // produces the type of a deref without worrying about whether a
    // move out would actually be legal
    loop { }
}

fn load0(ss: &MyBox<SomeTrait>) -> MyBox<SomeTrait> {
    deref(ss) //~ ERROR cannot infer
}

fn load1<'a,'b>(a: &'a MyBox<SomeTrait>,
                b: &'b MyBox<SomeTrait>)
                -> &'b MyBox<SomeTrait>
{
    a
      //~^ ERROR cannot infer
      //~| ERROR mismatched types
}

fn main() {
}
