// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


trait HasNumber<T> {
    const Number: usize;
}

enum One {}
enum Two {}

enum Foo {}

impl<T> HasNumber<T> for One {
    const Number: usize = 1;
}

impl<T> HasNumber<T> for Two {
    const Number: usize = 2;
}

fn main() {}
