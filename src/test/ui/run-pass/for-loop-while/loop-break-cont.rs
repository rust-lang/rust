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
  let mut i = 0_usize;
  loop {
    println!("a");
    i += 1_usize;
    if i == 10_usize {
      break;
    }
  }
  assert_eq!(i, 10_usize);
  let mut is_even = false;
  loop {
    if i == 21_usize {
        break;
    }
    println!("b");
    is_even = false;
    i += 1_usize;
    if i % 2_usize != 0_usize {
        continue;
    }
    is_even = true;
  }
  assert!(!is_even);
  loop {
    println!("c");
    if i == 22_usize {
        break;
    }
    is_even = false;
    i += 1_usize;
    if i % 2_usize != 0_usize {
        continue;
    }
    is_even = true;
  }
  assert!(is_even);
}
