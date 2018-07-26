// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(extern_types)]

#[link(name = "ctest", kind = "static")]
extern {
    type data;

    fn data_create(magic: u32) -> *mut data;
    fn data_get(data: *mut data) -> u32;
}

const MAGIC: u32 = 0xdeadbeef;
fn main() {
    unsafe {
        let data = data_create(MAGIC);
        assert_eq!(data_get(data), MAGIC);
    }
}
