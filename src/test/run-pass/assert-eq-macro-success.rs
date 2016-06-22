// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(PartialEq, Debug)]
struct Point { x : isize }

pub fn main() {
    assert_eq!(14,14);
    assert_eq!("abc".to_string(),"abc".to_string());
    assert_eq!(Box::new(Point{x:34}),Box::new(Point{x:34}));
    assert_eq!(&Point{x:34},&Point{x:34});
    assert_eq!(42, 42, "foo bar");
    assert_eq!(42, 42, "a {} c", "b");
    assert_eq!(42, 42, "{x}, {y}, {z}", x = 1, y = 2, z = 3);
}
