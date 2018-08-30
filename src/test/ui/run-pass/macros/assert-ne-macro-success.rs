// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
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
    assert_ne!(666,14);
    assert_ne!("666".to_string(),"abc".to_string());
    assert_ne!(Box::new(Point{x:666}),Box::new(Point{x:34}));
    assert_ne!(&Point{x:666},&Point{x:34});
    assert_ne!(666, 42, "no gods no masters");
    assert_ne!(666, 42, "6 {} 6", "6");
    assert_ne!(666, 42, "{x}, {y}, {z}", x = 6, y = 6, z = 6);
}
