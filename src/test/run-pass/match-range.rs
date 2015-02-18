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
    match 5_usize {
      1_usize...5_usize => {}
      _ => panic!("should match range"),
    }
    match 5_usize {
      6_usize...7_usize => panic!("shouldn't match range"),
      _ => {}
    }
    match 5_usize {
      1_usize => panic!("should match non-first range"),
      2_usize...6_usize => {}
      _ => panic!("math is broken")
    }
    match 'c' {
      'a'...'z' => {}
      _ => panic!("should suppport char ranges")
    }
    match -3 {
      -7...5 => {}
      _ => panic!("should match signed range")
    }
    match 3.0f64 {
      1.0...5.0 => {}
      _ => panic!("should match float range")
    }
    match -1.5f64 {
      -3.6...3.6 => {}
      _ => panic!("should match negative float range")
    }
}
