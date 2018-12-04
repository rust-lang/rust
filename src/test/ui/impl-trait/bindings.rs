// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(impl_trait_in_bindings)]

fn a<T: Clone>(x: T) {
    const foo: impl Clone = x;
//~^ ERROR can't capture dynamic environment in a fn item
}

fn b<T: Clone>(x: T) {
    let _ = move || {
        const foo: impl Clone = x;
//~^ ERROR can't capture dynamic environment in a fn item
    };
}

trait Foo<T: Clone> {
    fn a(x: T) {
        const foo: impl Clone = x;
//~^ ERROR can't capture dynamic environment in a fn item
    }
}

impl<T: Clone> Foo<T> for i32 {
    fn a(x: T) {
        const foo: impl Clone = x;
//~^ ERROR can't capture dynamic environment in a fn item
    }
}

fn main() { }
