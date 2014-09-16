// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


fn f<T>(x: Vec<T>) -> T { return x.into_iter().next().unwrap(); }

fn g(act: |Vec<int> | -> int) -> int { return act(vec!(1, 2, 3)); }

pub fn main() {
    assert_eq!(g(f), 1);
    let f1: |Vec<String>| -> String = f;
    assert_eq!(f1(vec!["x".to_string(), "y".to_string(), "z".to_string()]),
               "x".to_string());
}
