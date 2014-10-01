// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15877

pub fn main() {
    match 5u {
      1u...5u => {}
      _ => fail!("should match range"),
    }
    match 5u {
      6u...7u => fail!("shouldn't match range"),
      _ => {}
    }
    match 5u {
      1u => fail!("should match non-first range"),
      2u...6u => {}
      _ => fail!("math is broken")
    }
    match 'c' {
      'a'...'z' => {}
      _ => fail!("should suppport char ranges")
    }
    match -3i {
      -7...5 => {}
      _ => fail!("should match signed range")
    }
    match 3.0f64 {
      1.0...5.0 => {}
      _ => fail!("should match float range")
    }
    match -1.5f64 {
      -3.6...3.6 => {}
      _ => fail!("should match negative float range")
    }
}
