// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#[deriving(PartialEq, Show)]
struct Point { x : int }

pub fn main() {
    assert_eq!(14i,14i);
    assert_eq!("abc".to_string(),"abc".to_string());
    assert_eq!(box Point{x:34},box Point{x:34});
    assert_eq!(&Point{x:34},&Point{x:34});
}
