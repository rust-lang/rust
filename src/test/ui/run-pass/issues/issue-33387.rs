// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]

use std::sync::Arc;

trait Foo {
    fn get(&self) -> [u8; 2];
}

impl Foo for [u8; 2] {
    fn get(&self) -> [u8; 2] {
        *self
    }
}

struct Bar<T: ?Sized>(T);

fn unsize_fat_ptr<'a>(x: &'a Bar<Foo + Send + 'a>) -> &'a Bar<Foo + 'a> {
    x
}

fn unsize_nested_fat_ptr(x: Arc<Foo + Send>) -> Arc<Foo> {
    x
}

fn main() {
    let x: Box<Bar<Foo + Send>> = Box::new(Bar([1,2]));
    assert_eq!(unsize_fat_ptr(&*x).0.get(), [1, 2]);

    let x: Arc<Foo + Send> = Arc::new([3, 4]);
    assert_eq!(unsize_nested_fat_ptr(x).get(), [3, 4]);
}
