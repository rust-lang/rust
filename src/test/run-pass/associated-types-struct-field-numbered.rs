// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we correctly normalize the type of a struct field
// which has an associated type.

pub trait UnifyKey {
    type Value;

    fn dummy(&self) { }
}

pub struct Node<K:UnifyKey>(K, K::Value);

fn foo<K : UnifyKey<Value=Option<V>>,V : Clone>(node: &Node<K>) -> Option<V> {
    node.1.clone()
}

impl UnifyKey for i32 {
    type Value = Option<u32>;
}

impl UnifyKey for u32 {
    type Value = Option<i32>;
}

pub fn main() {
    let node: Node<i32> = Node(1, Some(22));
    assert_eq!(foo(&node), Some(22_u32));

    let node: Node<u32> = Node(1, Some(22));
    assert_eq!(foo(&node), Some(22_i32));
}
