//@ known-bug: #132826
pub trait MyTrait {
    type Item;
}

impl<K> MyTrait for Vec<K> {
    type Item = Vec<K>;
}

impl<K> From<Vec<K>> for <Vec<K> as MyTrait>::Item {}
