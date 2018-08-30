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
    let x = vec![1, 2, 3];
    let mut y = 0;
    for i in &x { println!("{}", *i); y += *i; }
    println!("{}", y);
    assert_eq!(y, 6);
    let s = "hello there".to_string();
    let mut i: isize = 0;
    for c in s.bytes() {
        if i == 0 { assert_eq!(c, 'h' as u8); }
        if i == 1 { assert_eq!(c, 'e' as u8); }
        if i == 2 { assert_eq!(c, 'l' as u8); }
        if i == 3 { assert_eq!(c, 'l' as u8); }
        if i == 4 { assert_eq!(c, 'o' as u8); }
        // ...

        i += 1;
        println!("{}", i);
        println!("{}", c);
    }
    assert_eq!(i, 11);
}
