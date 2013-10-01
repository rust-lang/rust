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
    match 5u {
      1u..5u => {}
      _ => fail2!("should match range"),
    }
    match 5u {
      6u..7u => fail2!("shouldn't match range"),
      _ => {}
    }
    match 5u {
      1u => fail2!("should match non-first range"),
      2u..6u => {}
      _ => fail2!("math is broken")
    }
    match 'c' {
      'a'..'z' => {}
      _ => fail2!("should suppport char ranges")
    }
    match -3 {
      -7..5 => {}
      _ => fail2!("should match signed range")
    }
    match 3.0 {
      1.0..5.0 => {}
      _ => fail2!("should match float range")
    }
    match -1.5 {
      -3.6..3.6 => {}
      _ => fail2!("should match negative float range")
    }
}
