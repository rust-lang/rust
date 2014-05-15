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
    let v: Vec<int> = vec!(0, 1, 2, 3, 4, 5);
    let s: StrBuf = "abcdef".to_strbuf();
    assert_eq!(v.as_slice()[3u], 3);
    assert_eq!(v.as_slice()[3u8], 3); //~ ERROR: mismatched types
    assert_eq!(v.as_slice()[3i8], 3); //~ ERROR: mismatched types
    assert_eq!(v.as_slice()[3u32], 3); //~ ERROR: mismatched types
    assert_eq!(v.as_slice()[3i32], 3); //~ ERROR: mismatched types
    println!("{}", v.as_slice()[3u8]); //~ ERROR: mismatched types
    assert_eq!(s.as_slice()[3u], 'd' as u8);
    assert_eq!(s.as_slice()[3u8], 'd' as u8); //~ ERROR: mismatched types
    assert_eq!(s.as_slice()[3i8], 'd' as u8); //~ ERROR: mismatched types
    assert_eq!(s.as_slice()[3u32], 'd' as u8); //~ ERROR: mismatched types
    assert_eq!(s.as_slice()[3i32], 'd' as u8); //~ ERROR: mismatched types
    println!("{}", s.as_slice()[3u8]); //~ ERROR: mismatched types
}
