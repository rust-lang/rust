// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Stringify {
    fn to_string(&self) -> String;
}

impl Stringify for u32 {
    fn to_string(&self) -> String { format!("u32: {}", *self) }
}

impl Stringify for f32 {
    fn to_string(&self) -> String { format!("f32: {}", *self) }
}

fn print<T: Stringify>(x: T) -> String {
    x.to_string()
}

fn main() {
    assert_eq!(&print(5), "u32: 5");
    assert_eq!(&print(5.0), "f32: 5");
}
