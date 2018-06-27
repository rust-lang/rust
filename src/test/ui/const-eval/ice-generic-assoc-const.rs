// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

pub trait Nullable {
    const NULL: Self;

    fn is_null(&self) -> bool;
}

impl<T> Nullable for *const T {
    const NULL: Self = 0 as *const T;

    fn is_null(&self) -> bool {
        *self == Self::NULL
    }
}

fn main() {
}
