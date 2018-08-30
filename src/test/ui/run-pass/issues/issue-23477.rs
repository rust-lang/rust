// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
// compiler-flags: -g

pub struct Dst {
    pub a: (),
    pub b: (),
    pub data: [u8],
}

pub unsafe fn borrow(bytes: &[u8]) -> &Dst {
    let dst: &Dst = std::mem::transmute((bytes.as_ptr(), bytes.len()));
    dst
}

fn main() {}
