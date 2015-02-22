// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn foo1<T:Copy<U>, U>(x: T) {}
//~^ ERROR: builtin bounds do not require arguments, 1 given

trait Trait: Copy<Send> {}
//~^ ERROR: builtin bounds do not require arguments, 1 given

struct MyStruct1<T: Copy<T>>;
//~^ ERROR builtin bounds do not require arguments, 1 given

struct MyStruct2<'a, T: Copy<'a>>;
//~^ ERROR: builtin bounds do not require arguments, 1 given

fn foo2<'a, T:Copy<'a, U>, U>(x: T) {}
//~^ ERROR builtin bounds do not require arguments, 2 given

fn main() {
}
