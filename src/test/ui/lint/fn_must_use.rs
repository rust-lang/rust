// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(fn_must_use)]
#![warn(unused_must_use)]

struct MyStruct {
    n: usize
}

impl MyStruct {
    #[must_use]
    fn need_to_use_this_method_value(&self) -> usize {
        self.n
    }
}

#[must_use="it's important"]
fn need_to_use_this_value() -> bool {
    false
}

fn main() {
    need_to_use_this_value();

    let m = MyStruct { n: 2 };
    m.need_to_use_this_method_value();
}
