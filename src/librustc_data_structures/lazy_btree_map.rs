// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::btree_map;
use std::collections::BTreeMap;

/// A thin wrapper around BTreeMap that avoids allocating upon creation.
///
/// Vec, HashSet and HashMap all have the nice feature that they don't do any
/// heap allocation when creating a new structure of the default size. In
/// contrast, BTreeMap *does* allocate in that situation. The compiler uses
/// B-Tree maps in some places such that many maps are created but few are
/// inserted into, so having a BTreeMap alternative that avoids allocating on
/// creation is a performance win.
///
/// Only a fraction of BTreeMap's functionality is currently supported.
/// Additional functionality should be added on demand.
#[derive(Debug)]
pub struct LazyBTreeMap<K, V>(Option<BTreeMap<K, V>>);

impl<K, V> LazyBTreeMap<K, V> {
    pub fn new() -> LazyBTreeMap<K, V> {
        LazyBTreeMap(None)
    }

    pub fn iter(&self) -> Iter<K, V> {
        Iter(self.0.as_ref().map(|btm| btm.iter()))
    }

    pub fn is_empty(&self) -> bool {
        self.0.as_ref().map_or(true, |btm| btm.is_empty())
    }
}

impl<K: Ord, V> LazyBTreeMap<K, V> {
    fn instantiate(&mut self) -> &mut BTreeMap<K, V> {
        if let Some(ref mut btm) = self.0 {
            btm
        } else {
            let btm = BTreeMap::new();
            self.0 = Some(btm);
            self.0.as_mut().unwrap()
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.instantiate().insert(key, value)
    }

    pub fn entry(&mut self, key: K) -> btree_map::Entry<K, V> {
        self.instantiate().entry(key)
    }

    pub fn values<'a>(&'a self) -> Values<'a, K, V> {
        Values(self.0.as_ref().map(|btm| btm.values()))
    }
}

impl<K: Ord, V> Default for LazyBTreeMap<K, V> {
    fn default() -> LazyBTreeMap<K, V> {
        LazyBTreeMap::new()
    }
}

impl<'a, K: 'a, V: 'a> IntoIterator for &'a LazyBTreeMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}

pub struct Iter<'a, K: 'a, V: 'a>(Option<btree_map::Iter<'a, K, V>>);

impl<'a, K: 'a, V: 'a> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        self.0.as_mut().and_then(|iter| iter.next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.as_ref().map_or_else(|| (0, Some(0)), |iter| iter.size_hint())
    }
}

pub struct Values<'a, K: 'a, V: 'a>(Option<btree_map::Values<'a, K, V>>);

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<&'a V> {
        self.0.as_mut().and_then(|values| values.next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.as_ref().map_or_else(|| (0, Some(0)), |values| values.size_hint())
    }
}

