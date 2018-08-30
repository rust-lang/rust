// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #25436: permit token-trees to be followed
// by sequences, enabling more general parsing.

use self::Join::*;

#[derive(Debug)]
enum Join<A,B> {
  Keep(A,B),
  Skip(A,B),
}

macro_rules! parse_list {
  ( < $a:expr; > $($b:tt)* ) => { Keep(parse_item!($a),parse_list!($($b)*)) };
  ( $a:tt $($b:tt)* ) => { Skip(parse_item!($a), parse_list!($($b)*)) };
  ( ) => { () };
}

macro_rules! parse_item {
  ( $x:expr ) => { $x }
}

fn main() {
    let list = parse_list!(<1;> 2 <3;> 4);
    assert_eq!("Keep(1, Skip(2, Keep(3, Skip(4, ()))))",
               format!("{:?}", list));
}
