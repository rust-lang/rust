//@ check-pass

use std::collections::{BTreeMap, HashMap};

trait Map
where
    for<'a> &'a Self: IntoIterator<Item = (&'a Self::Key, &'a Self::Value)>,
{
    type Key;
    type Value;
}

impl<K, V> Map for HashMap<K, V> {
    type Key = K;
    type Value = V;
}

impl<K, V> Map for BTreeMap<K, V> {
  type Key = K;
  type Value = V;
}

fn main() {}
