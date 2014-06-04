// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::iter;
use core::mem;

pub type MoveItems<K, V> = vec::MoveItems<(K, V)>;
pub type Items<'a, K, V> =
    iter::Map<'static, &'a (K, V), (&'a K, &'a V), slice::Items<'a, (K, V)>>;
pub type MutItems<'a, K, V> =
    iter::Map<'static, &'a mut (K, V), (&'a K, &'a mut V), slice::MutItems<'a, (K, V)>>;

/// A simple map that only requires Eq on keys. Based on a vector of key/value
/// pairs, it requires O(number of elements) space and time for all operations.
#[deriving(Clone)]
pub struct SmallMap<K, V>{ v: Vec<(K, V)> }

impl<K: Eq, V> SmallMap<K, V> {
    pub fn new() -> SmallMap<K, V> { SmallMap{ v: Vec::new() } }

    pub fn move_iter(self) -> MoveItems<K, V> {
      self.v.move_iter()
    }

    pub fn iter<'a>(&'a self) -> Items<'a, K, V> {
      self.v.iter().map(|&(ref k, ref v)| (k, v))
    }

    pub fn mut_iter<'a>(&'a mut self) -> MutItems<'a, K, V> {
      self.v.mut_iter().map(|&(ref k, ref mut v)| (k, v))
    }
}

impl<K: Eq, V> Container for SmallMap<K, V> {
    fn len(&self) -> uint {
        self.v.len()
    }
}

impl<K: Eq, V> Map<K, V> for SmallMap<K, V> {
    fn find<'a>(&'a self, key: &K) -> Option<&'a V> {
        for (ekey, evalue) in self.iter() {
            if ekey == key {
                return Some(evalue);
            }
        }
        None
    }
}

impl<K: Eq, V> Mutable for SmallMap<K, V> {
    fn clear(&mut self) {
        self.v.clear()
    }
}

impl<K: Eq, V> MutableMap<K, V> for SmallMap<K, V> {
    fn swap(&mut self, key: K, value: V) -> Option<V> {
        for (ekey, evalue) in self.mut_iter() {
            if ekey == &key {
                return Some(mem::replace(evalue, value));
            }
        }
        self.v.push((key, value));
        None
    }

    fn pop(&mut self, key: &K) -> Option<V> {
        match self.iter().position(|(ekey, _)| ekey == key) {
            Some(i) => self.v.swap_remove(i).map(|(_, value)| value),
            None => None,
        }
    }

    fn find_mut<'a>(&'a mut self, key: &K) -> Option<&'a mut V> {
        for (ekey, evalue) in self.mut_iter() {
            if ekey == key {
                return Some(evalue);
            }
        }
        None
    }
}

impl<K: Eq, V> FromIterator<(K, V)> for SmallMap<K, V> {
    fn from_iter<T: Iterator<(K, V)>>(iter: T) -> SmallMap<K, V> {
        let mut map = SmallMap::new();
        map.extend(iter);
        map
    }
}

impl<K: Eq, V> Extendable<(K, V)> for SmallMap<K, V> {
    fn extend<T: Iterator<(K, V)>>(&mut self, mut iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

#[cfg(test)]
mod test_map {
    use super::SmallMap;

    #[test]
    fn test_general() {
        let mut m = SmallMap::new();
        assert_eq!(m.len(), 0);
        assert_eq!(m.iter().next(), None);

        assert!(!m.contains_key(&4));
        assert!(m.insert(4, 5));
        assert_eq!(m.find(&4), Some(&5));
        assert_eq!(m.len(), 1);

        assert!(!m.insert(4, 6));
        assert_eq!(m.find(&4), Some(&6));
        assert_eq!(m.len(), 1);

        assert_eq!(m.find(&5), None);
        assert!(m.insert(5, 7));
        assert_eq!(m.find(&4), Some(&6));
        assert_eq!(m.find(&5), Some(&7));

        *m.find_mut(&5).unwrap() = 8;
        assert_eq!(m.find(&5), Some(&8));

        let mut elems = m.move_iter().collect::<Vec<(int, int)>>();
        elems.sort();
        assert_eq!(elems, vec![(4, 6), (5, 8)]);
    }

    #[test]
    fn test_collect_swap_pop() {
        let v = vec![(5, 8), (4, 6)];
        let mut m = v.move_iter().collect::<SmallMap<int, int>>();
        assert_eq!(m.len(), 2);
        assert_eq!(m.find(&4), Some(&6));
        assert_eq!(m.find(&5), Some(&8));

        m.clear();
        assert_eq!(m.len(), 0);
        assert_eq!(m.iter().next(), None);
        assert!(!m.contains_key(&4));
        assert!(!m.contains_key(&5));

        assert_eq!(m.swap(4, 9), None);
        assert_eq!(m.find(&4), Some(&9));
        assert_eq!(m.swap(4, 0), Some(9));
        assert_eq!(m.find(&4), Some(&0));

        assert!(m.insert(5, 1));

        assert_eq!(m.pop(&4), Some(0));
        assert!(!m.contains_key(&4));
        assert_eq!(m.len(), 1);
        assert_eq!(m.find(&5), Some(&1));
    }
}
