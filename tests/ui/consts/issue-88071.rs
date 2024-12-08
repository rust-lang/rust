//@ check-pass
//
// regression test for #88071

use std::collections::BTreeMap;

pub struct CustomMap<K, V>(BTreeMap<K, V>);

impl<K, V> CustomMap<K, V>
where
    K: Ord,
{
    pub const fn new() -> Self {
        CustomMap(BTreeMap::new())
    }
}

fn main() {}
