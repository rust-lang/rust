// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.




// This is a testcase for issue #94.
pub fn main() {
    let v: ~[int] = ~[0, 1, 2, 3, 4, 5];
    let s: ~str = ~"abcdef";
    assert_eq!(v[3u], 3);
    assert_eq!(v[3u8], 3);
    assert_eq!(v[3i8], 3);
    assert_eq!(v[3u32], 3);
    assert_eq!(v[3i32], 3);
    info2!("{}", v[3u8]);
    assert_eq!(s[3u], 'd' as u8);
    assert_eq!(s[3u8], 'd' as u8);
    assert_eq!(s[3i8], 'd' as u8);
    assert_eq!(s[3u32], 'd' as u8);
    assert_eq!(s[3i32], 'd' as u8);
    info2!("{}", s[3u8]);
}
