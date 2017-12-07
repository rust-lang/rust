// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(const_fn)]
#![warn(const_err)]

pub struct Data<T> {
    function: fn() -> T,
}

impl<T> Data<T> {
    pub const fn new(function: fn() -> T) -> Data<T> {
        Data {
            function: function,
        }
    }
}

pub static DATA: Data<i32> = Data::new(|| {
    413i32
});

fn main() {
    print!("{:?}", (DATA.function)());
}
