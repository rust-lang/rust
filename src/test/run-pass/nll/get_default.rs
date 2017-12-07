// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(nll)]

use std::collections::HashMap;

fn get_default(map: &mut HashMap<usize, String>, key: usize) -> &mut String {
    match map.get_mut(&key) {
        Some(value) => value,
        None => {
            map.insert(key, "".to_string());
            map.get_mut(&key).unwrap()
        }
    }
}

fn main() {
    let map = &mut HashMap::new();
    map.insert(22, format!("Hello, world"));
    map.insert(44, format!("Goodbye, world"));
    assert_eq!(&*get_default(map, 22), "Hello, world");
    assert_eq!(&*get_default(map, 66), "");
}
