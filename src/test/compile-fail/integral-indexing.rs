// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    let v: Vec<isize> = vec![0, 1, 2, 3, 4, 5];
    let s: String = "abcdef".to_string();
    v[3_usize];
    v[3];
    v[3u8];  //~ERROR : std::ops::Index<u8>` is not satisfied
    v[3i8];  //~ERROR : std::ops::Index<i8>` is not satisfied
    v[3u32]; //~ERROR : std::ops::Index<u32>` is not satisfied
    v[3i32]; //~ERROR : std::ops::Index<i32>` is not satisfied
    s.as_bytes()[3_usize];
    s.as_bytes()[3];
    s.as_bytes()[3u8];  //~ERROR : std::slice::SliceIndex<u8>` is not satisfied
    s.as_bytes()[3i8];  //~ERROR : std::slice::SliceIndex<u8>` is not satisfied
    s.as_bytes()[3u32]; //~ERROR : std::slice::SliceIndex<u8>` is not satisfied
    s.as_bytes()[3i32]; //~ERROR : std::slice::SliceIndex<u8>` is not satisfied
}
