// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Field/method access is parsed after type ascription or cast expressions

#![feature(type_ascription)]

trait Tr {
    fn s(&self) -> String;
}

impl<T: std::fmt::Display> Tr for T {
    fn s(&self) -> String {
        self.to_string()
    }
}

fn main() {
    let a: u8 = 10;
    let b = a as u16.s();
    let c = a: u8.s();
    let d = 11: u8.to_string();
    assert_eq!(b, "10");
    assert_eq!(c, "10");
    assert_eq!(d, "11");

    let v = [1, 2, 3];
    let len = v.iter().
                collect(): Vec<_>.
                len();
    assert_eq!(len, 3);
}
