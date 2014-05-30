// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test overloading of the `[]` operator.  In particular test that it
// takes its argument *by reference*.

extern crate debug;

use std::ops::Index;

struct AssociationList<K,V> {
    pairs: Vec<AssociationPair<K,V>> }

#[deriving(Clone)]
struct AssociationPair<K,V> {
    key: K,
    value: V
}

impl<K,V> AssociationList<K,V> {
    fn push(&mut self, key: K, value: V) {
        self.pairs.push(AssociationPair {key: key, value: value});
    }
}

impl<K:PartialEq,V:Clone> Index<K,V> for AssociationList<K,V> {
    fn index(&self, index: &K) -> V {
        for pair in self.pairs.iter() {
            if pair.key == *index {
                return pair.value.clone();
            }
        }
        fail!("No value found for key: {:?}", index);
    }
}

pub fn main() {
    let foo = "foo".to_string();
    let bar = "bar".to_string();

    let mut list = AssociationList {pairs: Vec::new()};
    list.push(foo.clone(), 22);
    list.push(bar.clone(), 44);

    assert!(list[foo] == 22)
    assert!(list[bar] == 44)

    assert!(list[foo] == 22)
    assert!(list[bar] == 44)
}
