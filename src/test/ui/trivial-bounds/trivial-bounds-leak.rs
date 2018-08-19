// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that false bounds don't leak
#![feature(trivial_bounds)]

pub trait Foo {
    fn test(&self);
}

fn return_str() -> str where str: Sized {
    *"Sized".to_string().into_boxed_str()
}

fn cant_return_str() -> str { //~ ERROR
    *"Sized".to_string().into_boxed_str()
}

fn my_function() where i32: Foo
{
    3i32.test();
    Foo::test(&4i32);
    generic_function(5i32);
}

fn foo() {
    3i32.test(); //~ ERROR
    Foo::test(&4i32); //~ ERROR
    generic_function(5i32); //~ ERROR
}

fn generic_function<T: Foo>(t: T) {}

fn main() {}

