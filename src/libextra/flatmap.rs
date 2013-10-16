// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A mapping type stored in a vector [(Key, Value)]

extern mod extra;

use std::vec;
use std::util::replace;
use std::cast;

// TODO: FlatSet<K>

#[inline]
fn lower_bound_index<K: TotalOrd, V>(a: &[(K,V)], key: &K) -> uint {
    let mut count = a.len();
    let mut first = 0u;
    let mut it;
    let mut step;
    while count > 0 {
        it = first;
        step = count / 2;
        it += step;
        let (ref k, _) = a[it];
        if k.cmp(key) == Less {
            first = it + 1;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    first
}

pub struct FlatMap<K, V> {
    priv data: ~[(K, V)],
}

impl<K: TotalOrd, V> FlatMap<K, V> {
    /// Creates an empty FlatMap.
    fn new() -> FlatMap<K, V> {
        FlatMap::with_capacity(2)
    }

    /// Create an empty FlatMap with space for at least `n` elements.
    fn with_capacity(capacity: uint) -> FlatMap<K, V> {
        FlatMap{data: vec::with_capacity(capacity)}
    }

    /// Return the capacity of the underlying vector.
    fn capacity(&self) -> uint {
        self.data.capacity()
    }

    /// Modify and return the value corresponding to the key in the map, or
    /// insert and return a new value if it doesn't exist.
    pub fn mangle<'a,A>(&'a mut self, k: K, a: A, not_found: &fn(&K, A) -> V,
                        found: &fn(&K, &mut V, A)) -> &'a mut V {
        // work around https://github.com/mozilla/rust/issues/6393
        let self2: &mut FlatMap<K, V> = unsafe {
            cast::transmute_copy(&self)
        };
        match self.find_mut(&k) {
            Some(val) => {
                found(&k, val, a);
                val
            }
            None => {
                let v = not_found(&k, a);
                let index = lower_bound_index(self2.data, &k);
                self2.data.insert(index, (k, v));
                let (_, ref val) = self2.data[index];
                unsafe { cast::transmute(val) }
            }
        }
    }

    /// Return the value corresponding to the key in the map, or insert
    /// and return the value if it doesn't exist.
    pub fn find_or_insert<'a>(&'a mut self, k: K, v: V) -> &'a mut V {
        self.mangle(k, v, |_k, a| a, |_k,_v,_a| ())
    }

    /// Return the value corresponding to the key in the map, or create,
    /// insert, and return a new value if it doesn't exist.
    pub fn find_or_insert_with<'a>(&'a mut self, k: K, f: &fn(&K) -> V)
                               -> &'a mut V {
        self.mangle(k, (), |k,_a| f(k), |_k,_v,_a| ())
    }

    /// Insert a key-value pair into the map if the key is not already present.
    /// Otherwise, modify the existing value for the key.
    /// Returns the new or modified value for the key.
    pub fn insert_or_update_with<'a>(&'a mut self, k: K, v: V,
                                     f: &fn(&K, &mut V)) -> &'a mut V {
        self.mangle(k, v, |_k,a| a, |k,v,_a| f(k,v))
    }

    /// Retrieves a value for the given key, failing if the key is not
    /// present.
    pub fn get<'a>(&'a self, k: &K) -> &'a V {
        match self.find(k) {
            Some(v) => v,
            None => fail!("No entry found for key: %?", k),
        }
    }

    /// Retrieves a (mutable) value for the given key, failing if the key
    /// is not present.
    pub fn get_mut<'a>(&'a mut self, k: &K) -> &'a mut V {
        match self.find_mut(k) {
            Some(v) => v,
            None => fail!("No entry found for key: %?", k),
        }
    }

    /// Return the value corresponding to the key in the map, using
    /// equivalence
    pub fn find_equiv<'a, Q: Equiv<K>>(&'a self, key: &Q)
                                             -> Option<&'a V> {
        for &(ref k, ref v) in self.data.iter() {
            if key.equiv(k) {
                return Some(v)
            }
        }
        None
    }

    /// An iterator visiting all key-value pairs in order.
    /// Iterator element type is (&'a K, &'a V).
    pub fn iter<'a>(&'a self) -> FlatMapIterator<'a, K, V> {
        FlatMapIterator{iter: self.data.iter()}
    }

    /// An iterator visiting all key-value pairs in order,
    /// with mutable references to the values.
    /// Iterator element type is (&'a K, &'a mut V).
    fn mut_iter<'a>(&'a mut self) -> FlatMapMutIterator<'a, K, V> {
        FlatMapMutIterator{iter: self.data.mut_iter()}
    }

    /// Creates a consuming iterator, that is, one that moves each key-value
    /// pair out of the map in reverse order. The map cannot be used after
    /// calling this.
    pub fn move_iter(self) -> FlatMapMoveIterator<K, V> {
        // `move_rev_iter` is more efficient than `move_iter` for vectors
        FlatMapMoveIterator {iter: self.data.move_rev_iter()}
    }
}

impl<K, V> Container for FlatMap<K, V> {
    /// Return the number of elements in the map.
    fn len(&self) -> uint {
        self.data.len()
    }
}

impl<K, V> Mutable for FlatMap<K, V> {
    /// Clear the map, removing all items.
    fn clear(&mut self) {
        self.data.clear();
    }
}

impl<K: TotalOrd, V> FromIterator<(K, V)> for FlatMap<K, V> {
    fn from_iterator<T: Iterator<(K, V)>>(iter: &mut T) -> FlatMap<K, V> {
        let (lower, _) = iter.size_hint();
        let mut map = FlatMap::with_capacity(lower);
        map.extend(iter);
        map
    }
}

impl<K: TotalOrd, V> Extendable<(K, V)> for FlatMap<K, V> {
    fn extend<T: Iterator<(K, V)>>(&mut self, iter: &mut T) {
        for (k, v) in *iter {
            self.insert(k, v);
        }
    }
}

impl<K: TotalOrd, V> Default for FlatMap<K, V> {
    fn default() -> FlatMap<K, V> { FlatMap::new() }
}

impl<K: TotalOrd, V: Eq> Eq for FlatMap<K, V> {
    fn eq(&self, other: &FlatMap<K, V>) -> bool {
        if self.len() != other.len() { return false; }

        do self.iter().all |(key, value)| {
            match other.find(key) {
                None => false,
                Some(v) => value == v
            }
        }
    }
}

impl<K: TotalOrd, V> Map<K, V> for FlatMap<K, V> {
    fn find<'a>(&'a self, key: &K) -> Option<&'a V> {
        match self.data.bsearch(|&(ref k,_)|{k.cmp(key)}) {
            None => None,
            Some(idx) => {
                let (_, ref v) = self.data[idx];
                Some(v)
            }
        }
    }

    fn contains_key(&self, key: &K) -> bool {
        match self.find(key) {
            Some(_) => true,
            None => false
        }
    }
}

impl<K: TotalOrd, V> MutableMap<K, V> for FlatMap<K, V> {
    fn find_mut<'a>(&'a mut self, key: &K) -> Option<&'a mut V> {
        match self.data.bsearch(|&(ref k,_)|{k.cmp(key)}) {
            None => None,
            Some(idx) => {
                let (_, ref mut v) = self.data[idx];
                Some(v)
            }
        }
    }

    fn swap(&mut self, k: K, v: V) -> Option<V> {
        // work around https://github.com/mozilla/rust/issues/6393
        let self2: &mut FlatMap<K, V> = unsafe {
            cast::transmute_copy(&self)
        };
        match self.find_mut(&k) {
            Some(val) => {
                Some(replace(val, v))
            }
            None => {
                let index = lower_bound_index(self2.data, &k);
                self2.data.insert(index, (k, v));
                None
            }
        }
    }

    fn pop(&mut self, key: &K) -> Option<V> {
        let mut index: Option<uint> = None;
        for (i, &(ref k, _)) in self.data.mut_iter().enumerate() {
            if key.cmp(k) == Equal {
                index = Some(i);
                break;
            }
        }

        match index {
            Some(i) => {
                let (_, v) = self.data.remove(i);
                Some(v)
            }
            None => None
        }
    }
}

pub struct FlatMapIterator<'self, K, V> {
    priv iter: vec::VecIterator<'self, (K, V)>,
}

impl<'self, K, V> Iterator<(&'self K, &'self V)> for FlatMapIterator<'self, K, V> {
    #[inline]
    fn next(&mut self) -> Option<(&'self K, &'self V)> {
        match self.iter.next() {
            Some(&(ref k, ref v)) => Some((k, v)),
            None => None
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        return self.iter.size_hint();
    }
}

pub struct FlatMapMutIterator<'self, K, V> {
    priv iter: vec::VecMutIterator<'self, (K, V)>,
}

impl<'self, K, V> Iterator<(&'self K, &'self mut V)> for FlatMapMutIterator<'self, K, V> {
    #[inline]
    fn next(&mut self) -> Option<(&'self K, &'self mut V)> {
        match self.iter.next() {
            Some(&(ref k, ref mut v)) => Some((k, v)),
            None => None
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        return self.iter.size_hint();
    }
}

pub struct FlatMapMoveIterator<K, V> {
    priv iter: vec::MoveRevIterator<(K, V)>,
}

impl<K, V> Iterator<(K, V)> for FlatMapMoveIterator<K, V> {
    #[inline]
    fn next(&mut self) -> Option<(K, V)> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        return self.iter.size_hint();
    }
}

#[cfg(test)]
mod test {
    use super::FlatMap;

    #[test]
    fn test_container_trait() {
        let m: FlatMap<int, int> = FlatMap::with_capacity(10);
        assert!(10 == m.capacity());
        assert!(0 == m.len());
    }

    #[test]
    fn test_map_trait() {
        let m: FlatMap<int, int> = FlatMap::new();
        assert!(m.find(&0) == None);
        assert!(m.contains_key(&0) == false);
    }

    #[test]
    fn test_mutable_map_trait() {
        let mut m: FlatMap<int, int> = FlatMap::new();
        assert!(m.contains_key(&0) == false);
        m.insert(0, 1);
        assert!(m.contains_key(&0));
        m.insert(1, 2);
        assert!(m.len() == 2);
        assert!(m.contains_key(&1));
        m.pop(&0);
        assert!(m.contains_key(&0) == false);
        assert!(m.contains_key(&1));
        assert!(m.len() == 1);
        m.insert(1, 3);
        match m.find(&1) {
            Some(v) => assert_eq!(3, *v),
            None => fail!("No entry found for key: 1"),
        }

        for (k, v) in m.iter() {
            assert_eq!((1, 3), (*k, *v));
        }

        for (_, v) in m.mut_iter() {
            *v = 1;
        }

        for (k, v) in m.iter() {
            assert_eq!((1, 1), (*k, *v));
        }

        assert_eq!((1u, Some(1u)), m.iter().size_hint());
        m.insert(2, 2);
        m.insert(3, 3);
        assert_eq!((3u, Some(3u)), m.iter().size_hint());
    }
}

#[cfg(test)]
mod test_map {
    use super::FlatMap;

    #[test]
    fn test_create_capacity_zero() {
        let mut m = FlatMap::with_capacity(0);
        assert!(m.insert(1, 1));
    }

    #[test]
    fn test_insert() {
        let mut m = FlatMap::new();
        assert!(m.insert(1, 2));
        assert!(m.insert(2, 4));
        assert_eq!(*m.get(&1), 2);
        assert_eq!(*m.get(&2), 4);
    }

    #[test]
    fn test_find_mut() {
        let mut m = FlatMap::new();
        assert!(m.insert(1, 12));
        assert!(m.insert(2, 8));
        assert!(m.insert(5, 14));
        let new = 100;
        match m.find_mut(&5) {
            None => fail!(), Some(x) => *x = new
        }
        assert_eq!(m.find(&5), Some(&new));
    }

    #[test]
    fn test_insert_overwrite() {
        let mut m = FlatMap::new();
        assert!(m.insert(1, 2));
        assert_eq!(*m.get(&1), 2);
        assert!(!m.insert(1, 3));
        assert_eq!(*m.get(&1), 3);
    }

    #[test]
    fn test_insert_conflicts() {
        let mut m = FlatMap::with_capacity(4);
        assert!(m.insert(1, 2));
        assert!(m.insert(5, 3));
        assert!(m.insert(9, 4));
        assert_eq!(*m.get(&9), 4);
        assert_eq!(*m.get(&5), 3);
        assert_eq!(*m.get(&1), 2);
    }

    #[test]
    fn test_conflict_remove() {
        let mut m = FlatMap::with_capacity(4);
        assert!(m.insert(1, 2));
        assert!(m.insert(5, 3));
        assert!(m.insert(9, 4));
        assert!(m.remove(&1));
        assert_eq!(*m.get(&9), 4);
        assert_eq!(*m.get(&5), 3);
    }

    #[test]
    fn test_is_empty() {
        let mut m = FlatMap::with_capacity(4);
        assert!(m.insert(1, 2));
        assert!(!m.is_empty());
        assert!(m.remove(&1));
        assert!(m.is_empty());
    }

    #[test]
    fn test_pop() {
        let mut m = FlatMap::new();
        m.insert(1, 2);
        assert_eq!(m.pop(&1), Some(2));
        assert_eq!(m.pop(&1), None);
    }

    #[test]
    fn test_swap() {
        let mut m = FlatMap::new();
        assert_eq!(m.swap(1, 2), None);
        assert_eq!(m.swap(1, 3), Some(2));
        assert_eq!(m.swap(1, 4), Some(3));
    }

    #[test]
    fn test_find_or_insert() {
        let mut m: FlatMap<int,int> = FlatMap::new();
        assert_eq!(*m.find_or_insert(1, 2), 2);
        assert_eq!(*m.find_or_insert(1, 3), 2);
    }

    #[test]
    fn test_find_or_insert_with() {
        let mut m: FlatMap<int,int> = FlatMap::new();
        assert_eq!(*m.find_or_insert_with(1, |_| 2), 2);
        assert_eq!(*m.find_or_insert_with(1, |_| 3), 2);
    }

    #[test]
    fn test_insert_or_update_with() {
        let mut m: FlatMap<int,int> = FlatMap::new();
        assert_eq!(*m.insert_or_update_with(1, 2, |_,x| *x+=1), 2);
        assert_eq!(*m.insert_or_update_with(1, 2, |_,x| *x+=1), 3);
    }

    #[test]
    fn test_move_iter() {
        let hm = {
            let mut hm = FlatMap::new();

            hm.insert('a', 1);
            hm.insert('b', 2);

            hm
        };

        let v = hm.move_iter().collect::<~[(char, int)]>();
        assert!([('a', 1), ('b', 2)] == v || [('b', 2), ('a', 1)] == v);
    }

    #[test]
    fn test_iterate() {
        let mut m = FlatMap::with_capacity(4);
        for i in range(0u, 32) {
            assert!(m.insert(i, i*2));
        }
        let mut observed = 0;
        for (k, v) in m.iter() {
            assert_eq!(*v, *k * 2);
            observed |= (1 << *k);
        }
        assert_eq!(observed, 0xFFFF_FFFF);
    }

    #[test]
    fn test_find() {
        let mut m = FlatMap::new();
        assert!(m.find(&1).is_none());
        m.insert(1, 2);
        match m.find(&1) {
            None => fail!(),
            Some(v) => assert!(*v == 2)
        }
    }

    #[test]
    fn test_eq() {
        let mut m1 = FlatMap::new();
        m1.insert(1, 2);
        m1.insert(2, 3);
        m1.insert(3, 4);

        let mut m2 = FlatMap::new();
        m2.insert(1, 2);
        m2.insert(2, 3);

        assert!(m1 != m2);

        m2.insert(3, 4);

        assert_eq!(m1, m2);
    }

    #[test]
    fn test_find_equiv() {
        let mut m = FlatMap::new();

        let (foo, bar, baz) = (1,2,3);
        m.insert(~"foo", foo);
        m.insert(~"bar", bar);
        m.insert(~"baz", baz);


        assert_eq!(m.find_equiv(&("foo")), Some(&foo));
        assert_eq!(m.find_equiv(&("bar")), Some(&bar));
        assert_eq!(m.find_equiv(&("baz")), Some(&baz));

        assert_eq!(m.find_equiv(&("qux")), None);
    }

    #[test]
    fn test_from_iter() {
        let xs = ~[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: FlatMap<int, int> = xs.iter().map(|&x| x).collect();

        for &(k, v) in xs.iter() {
            assert_eq!(map.find(&k), Some(&v));
        }
    }
}


macro_rules! early_return(
    ($inp:expr $sp:ident) => ( // invoke it like `(input_5 special_e)`
        match $inp {
            $sp(x) => { return x; }
            _ => {}
        }
    );
)

macro_rules! bench_find {
    ($n:expr, $map_type:ident) => {{
        let size = $n;
        let mut m = $map_type::new();
        for i in range(0u, size) {
            assert!(m.insert(i, i));
        }
        let mut rng = rand::weak_rng();
        do bh.iter {
            let k: uint = rng.gen_integer_range(0u, size * 2);
            if k < size {
                assert!(m.find(&k).is_some());
            } else {
                assert!(m.find(&k).is_none());
            }
        }
    }}
}

macro_rules! bench_insert {
    ($n:expr, $map_type:ident) => {{
        do bh.iter {
            let mut m = $map_type::new();
            for i in range(0u, $n) {
                assert!(m.insert(i, i));
            }
        }
    }}
}

#[cfg(test)]
mod bench {
    use extra::test::BenchHarness;
    use super::FlatMap;
    use std::hashmap::HashMap;
    use extra::treemap::TreeMap;
    use std::rand;
    use std::rand::Rng;

    #[bench]
    fn hashmap_insert_10(bh: &mut BenchHarness) {
        bench_insert!(10, HashMap);
    }

    #[bench]
    fn hashmap_insert_100(bh: &mut BenchHarness) {
        bench_insert!(100, HashMap);
    }

    #[bench]
    fn hashmap_insert_1000(bh: &mut BenchHarness) {
        bench_insert!(1000, HashMap);
    }


    #[bench]
    fn treemap_insert_10(bh: &mut BenchHarness) {
        bench_insert!(10, TreeMap);
    }

    #[bench]
    fn treemap_insert_100(bh: &mut BenchHarness) {
        bench_insert!(100, TreeMap);
    }

    #[bench]
    fn treemap_insert_1000(bh: &mut BenchHarness) {
        bench_insert!(1000, TreeMap);
    }


    #[bench]
    fn flatmap_insert_10(bh: &mut BenchHarness) {
        bench_insert!(10, FlatMap);
    }

    #[bench]
    fn flatmap_insert_100(bh: &mut BenchHarness) {
        bench_insert!(100, FlatMap);
    }

    #[bench]
    fn flatmap_insert_1000(bh: &mut BenchHarness) {
        bench_insert!(1000, FlatMap);
    }


    #[bench]
    fn hashmap_find_10(bh: &mut BenchHarness) {
        bench_find!(10, HashMap);
    }

    #[bench]
    fn hashmap_find_100(bh: &mut BenchHarness) {
        bench_find!(100, HashMap);
    }

    #[bench]
    fn hashmap_find_1000(bh: &mut BenchHarness) {
        bench_find!(1000, HashMap);
    }


    #[bench]
    fn treemap_find_10(bh: &mut BenchHarness) {
        bench_find!(10, TreeMap);
    }

    #[bench]
    fn treemap_find_100(bh: &mut BenchHarness) {
        bench_find!(100, TreeMap);
    }

    #[bench]
    fn treemap_find_1000(bh: &mut BenchHarness) {
        bench_find!(1000, TreeMap);
    }


    #[bench]
    fn flatmap_find_10(bh: &mut BenchHarness) {
        bench_find!(10, FlatMap);
    }

    #[bench]
    fn flatmap_find_100(bh: &mut BenchHarness) {
        bench_find!(100, FlatMap);
    }

    #[bench]
    fn flatmap_find_1000(bh: &mut BenchHarness) {
        bench_find!(1000, FlatMap);
    }
}
