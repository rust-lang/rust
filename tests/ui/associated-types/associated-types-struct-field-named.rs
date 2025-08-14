//@ run-pass
// Test that we correctly normalize the type of a struct field
// which has an associated type.


pub trait UnifyKey {
    type Value;

    fn dummy(&self) { }
}

pub struct Node<K:UnifyKey> {
    pub key: K,
    pub value: K::Value,
}

fn foo<K : UnifyKey<Value=Option<V>>,V : Clone>(node: &Node<K>) -> Option<V> {
    node.value.clone()
}

impl UnifyKey for i32 {
    type Value = Option<u32>;
}

impl UnifyKey for u32 {
    type Value = Option<i32>;
}

pub fn main() {
    let node: Node<i32> = Node { key: 1, value: Some(22) };
    assert_eq!(foo(&node), Some(22));

    let node: Node<u32> = Node { key: 1, value: Some(22) };
    assert_eq!(foo(&node), Some(22));
}
