// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
// compile-flags: --error-format=human

#![feature(default_type_parameter_fallback)]

struct Bar<T>(T);

impl Bar<T> {
    fn new<U, Z:Default=String>() -> Bar<Z> {
        Bar(Z::default())
    }
}

fn main() {
    let _:u32 = foo::<usize>();
    let _:Bar<u32> = Bar::new::<usize>();
}

fn foo<U:Default, T:Default=String>() -> T {
    T::default()
}

