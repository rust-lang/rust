// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_features)]
#![feature(box_syntax)]

#[derive(Copy)]
struct LM { resize_at: uint, size: uint }

enum HashMap<K,V> {
    HashMap_(LM, Vec<(K,V)>)
}

fn linear_map<K,V>() -> HashMap<K,V> {
    HashMap::HashMap_(LM{
        resize_at: 32,
        size: 0}, Vec::new())
}

impl<K,V> HashMap<K,V> {
    pub fn len(&mut self) -> uint {
        match *self {
            HashMap::HashMap_(ref l, _) => l.size
        }
    }
}

pub fn main() {
    let mut m = box linear_map::<(),()>();
    assert_eq!(m.len(), 0);
}
