// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unsafe_destructor)]

// issue #20126

#[derive(Copy)] //~ ERROR the trait `Copy` may not be implemented
struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {}
}

#[derive(Copy)] //~ ERROR the trait `Copy` may not be implemented
struct Bar<T>(::std::marker::PhantomData<T>);

#[unsafe_destructor]
impl<T> Drop for Bar<T> {
    fn drop(&mut self) {}
}

fn main() {}
