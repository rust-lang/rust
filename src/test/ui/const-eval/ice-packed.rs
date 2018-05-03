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
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(packed)]
pub struct Num(u64);

impl Num {
    pub const ZERO: Self = Num(0);
}

pub fn decrement(a: Num) -> Num {
    match a {
        Num::ZERO => Num::ZERO,
        a => Num(a.0 - 1)
    }
}

fn main() {
}
