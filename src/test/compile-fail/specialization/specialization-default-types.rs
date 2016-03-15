// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// It should not be possible to use the concrete value of a defaulted
// associated type in the impl defining it -- otherwise, what happens
// if it's overridden?

#![feature(specialization)]

trait Example {
    type Output;
    fn generate(self) -> Self::Output;
}

impl<T> Example for T {
    default type Output = Box<T>;
    default fn generate(self) -> Self::Output {
        Box::new(self) //~ ERROR mismatched types
    }
}

impl Example for bool {
    type Output = bool;
    fn generate(self) -> bool { self }
}

fn trouble<T>(t: T) -> Box<T> {
    Example::generate(t) //~ ERROR mismatched types
}

fn weaponize() -> bool {
    let b: Box<bool> = trouble(true);
    *b
}

fn main() {
    weaponize();
}
