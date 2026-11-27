//@ run-pass
// Test overloading of the `[]` operator.  In particular test that it
// takes its argument *by reference*.

use std::ops::Index;

struct AssociationList<K,V> {
    pairs: Vec<AssociationPair<K,V>> }

#[derive(Clone)]
struct AssociationPair<K,V> {
    key: K,
    value: V
}

impl<K,V> AssociationList<K,V> {
    fn push(&mut self, key: K, value: V) {
        self.pairs.push(AssociationPair {key: key, value: value});
    }
}

impl<'a, K: PartialEq + std::fmt::Debug, V:Clone> Index<&'a K> for AssociationList<K,V> {
    type Output = V;

    fn index(&self, index: &K) -> &V {
        for pair in &self.pairs {
            if pair.key == *index {
                return &pair.value
            }
        }
        panic!("No value found for key: {:?}", index);
    }
}

pub fn main() {
    let foo = "foo".to_string();
    let bar = "bar".to_string();

    let mut list = AssociationList {pairs: Vec::new()};
    list.push(foo.clone(), 22);
    list.push(bar.clone(), 44);

    assert_eq!(list[&foo], 22);
    assert_eq!(list[&bar], 44);

    assert_eq!(list[&foo], 22);
    assert_eq!(list[&bar], 44);
}
