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

use std::ops::Index;

struct AssociationList<K,V> {
    pairs: ~[AssociationPair<K,V>]
}

#[deriving(Clone)]
struct MyVec<T> {
    values: ~[T]
}

impl<T> MyVec<T> {
    pub fn new() -> MyVec<T> {
      MyVec{ values: ~[] }
    }

    pub fn push(&mut self, val: T) {
        self.values.push(val);
    }
}

impl<T> IndexRef<int, T> for MyVec<T> {
    fn index<'a>(&'a self, index: &int) -> &'a T {
        &self.values[*index]
    }
}

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

impl<K:Eq,V:Clone> Index<K,V> for AssociationList<K,V> {
    fn index(&self, index: &K) -> V {
        for pair in self.pairs.iter() {
            if pair.key == *index {
                return pair.value.clone();
            }
        }
        fail!("No value found for key: {:?}", index);
    }
}

impl<K:Eq,V:Clone> IndexMut<K,V> for AssociationList<K,V> {
    fn index_mut<'a>(&'a mut self, index: &K) -> &'a mut V {
        for pair in self.pairs.mut_iter() {
            if pair.key == *index {
                &mut pair.value;
            }
        }
        fail!("No value found for key: {:?}", index);
    }
}

pub fn main() {
    let foo = ~"foo";
    let bar = ~"bar";

    let mut list = AssociationList {pairs: ~[]};
    list.push(foo.clone(), 22);
    list.push(bar.clone(), 44);

    assert!(list[foo] == 22)
    assert!(list[bar] == 44)

    assert!(list[foo] == 22)
    assert!(list[bar] == 44)

    let mut list = AssociationList { pairs: ~[] };
    list.push(foo.clone(), ~[1,2,3]);
    list.push(bar.clone(), ~[5,6,7]);

    assert_eq!(list[foo], ~[1,2,3]);
    assert_eq!(list[bar], ~[5,6,7]);

    let mut list = AssociationList{ pairs: ~[] };
    list.push(foo.clone(), MyVec::new());
    let mut my_vec = list[foo];
    my_vec.push(1);
    my_vec.push(2);
    assert_eq!(my_vec[0], &1);
    assert_eq!(my_vec[1], &2);
}
