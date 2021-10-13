// check-pass
//
// regression test for #88071

#![feature(const_btree_new)]
#![feature(const_fn_trait_bound)]

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
