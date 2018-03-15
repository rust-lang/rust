// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo {
    fn dummy(&self) { }
}

pub trait Bar : Foo {}
//~^ ERROR private trait `Foo` in public interface [E0445]
//~| NOTE private trait can't be public
pub struct Bar2<T: Foo>(pub T);
//~^ ERROR private trait `Foo` in public interface [E0445]
//~| NOTE private trait can't be public
pub fn foo<T: Foo> (t: T) {}
//~^ ERROR private trait `Foo` in public interface [E0445]
//~| NOTE private trait can't be public

fn main() {}
