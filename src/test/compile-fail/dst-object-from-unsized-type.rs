// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we cannot create objects from unsized types.

trait Foo { fn foo(&self) {} }
impl Foo for str {}
impl Foo for [u8] {}

fn test1<T: ?Sized + Foo>(t: &T) {
    let u: &Foo = t;
    //~^ ERROR the size for values of type
}

fn test2<T: ?Sized + Foo>(t: &T) {
    let v: &Foo = t as &Foo;
    //~^ ERROR the size for values of type
}

fn test3() {
    let _: &[&Foo] = &["hi"];
    //~^ ERROR the size for values of type
}

fn test4(x: &[u8]) {
    let _: &Foo = x as &Foo;
    //~^ ERROR the size for values of type
}

fn main() { }
