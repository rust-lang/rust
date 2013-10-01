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
  let mut i = 0u;
  loop {
    error2!("a");
    i += 1u;
    if i == 10u {
      break;
    }
  }
  assert_eq!(i, 10u);
  let mut is_even = false;
  loop {
    if i == 21u {
        break;
    }
    error2!("b");
    is_even = false;
    i += 1u;
    if i % 2u != 0u {
        loop;
    }
    is_even = true;
  }
  assert!(!is_even);
  loop {
    error2!("c");
    if i == 22u {
        break;
    }
    is_even = false;
    i += 1u;
    if i % 2u != 0u {
        loop;
    }
    is_even = true;
  }
  assert!(is_even);
}
