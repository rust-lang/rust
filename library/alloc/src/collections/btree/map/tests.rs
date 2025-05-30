use core::assert_matches::assert_matches;
use std::iter;
use std::ops::Bound::{Excluded, Included, Unbounded};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::SeqCst;

use super::*;
use crate::boxed::Box;
use crate::fmt::Debug;
use crate::rc::Rc;
use crate::string::{String, ToString};
use crate::testing::crash_test::{CrashTestDummy, Panic};
use crate::testing::ord_chaos::{Cyclic3, Governed, Governor};
use crate::testing::rng::DeterministicRng;

// Minimum number of elements to insert, to guarantee a tree with 2 levels,
// i.e., a tree who's root is an internal node at height 1, with edges to leaf nodes.
// It's not the minimum size: removing an element from such a tree does not always reduce height.
const MIN_INSERTS_HEIGHT_1: usize = node::CAPACITY + 1;

// Minimum number of elements to insert in ascending order, to guarantee a tree with 3 levels,
// i.e., a tree who's root is an internal node at height 2, with edges to more internal nodes.
// It's not the minimum size: removing an element from such a tree does not always reduce height.
const MIN_INSERTS_HEIGHT_2: usize = 89;

// Gathers all references from a mutable iterator and makes sure Miri notices if
// using them is dangerous.
fn test_all_refs<'a, T: 'a>(dummy: &mut T, iter: impl Iterator<Item = &'a mut T>) {
    // Gather all those references.
    let mut refs: Vec<&mut T> = iter.collect();
    // Use them all. Twice, to be sure we got all interleavings.
    for r in refs.iter_mut() {
        mem::swap(dummy, r);
    }
    for r in refs {
        mem::swap(dummy, r);
    }
}

impl<K, V> BTreeMap<K, V> {
    // Panics if the map (or the code navigating it) is corrupted.
    fn check_invariants(&self) {
        if let Some(root) = &self.root {
            let root_node = root.reborrow();

            // Check the back pointers top-down, before we attempt to rely on
            // more serious navigation code.
            assert!(root_node.ascend().is_err());
            root_node.assert_back_pointers();

            // Check consistency of `length` with what navigation code encounters.
            assert_eq!(self.length, root_node.calc_length());

            // Lastly, check the invariant causing the least harm.
            root_node.assert_min_len(if root_node.height() > 0 { 1 } else { 0 });
        } else {
            assert_eq!(self.length, 0);
        }

        // Check that `assert_strictly_ascending` will encounter all keys.
        assert_eq!(self.length, self.keys().count());
    }

    // Panics if the map is corrupted or if the keys are not in strictly
    // ascending order, in the current opinion of the `Ord` implementation.
    // If the `Ord` implementation violates transitivity, this method does not
    // guarantee that all keys are unique, just that adjacent keys are unique.
    fn check(&self)
    where
        K: Debug + Ord,
    {
        self.check_invariants();
        self.assert_strictly_ascending();
    }

    // Returns the height of the root, if any.
    fn height(&self) -> Option<usize> {
        self.root.as_ref().map(node::Root::height)
    }

    fn dump_keys(&self) -> String
    where
        K: Debug,
    {
        if let Some(root) = self.root.as_ref() {
            root.reborrow().dump_keys()
        } else {
            String::from("not yet allocated")
        }
    }

    // Panics if the keys are not in strictly ascending order.
    fn assert_strictly_ascending(&self)
    where
        K: Debug + Ord,
    {
        let mut keys = self.keys();
        if let Some(mut previous) = keys.next() {
            for next in keys {
                assert!(previous < next, "{:?} >= {:?}", previous, next);
                previous = next;
            }
        }
    }

    // Transform the tree to minimize wasted space, obtaining fewer nodes that
    // are mostly filled up to their capacity. The same compact tree could have
    // been obtained by inserting keys in a shrewd order.
    fn compact(&mut self)
    where
        K: Ord,
    {
        let iter = mem::take(self).into_iter();
        if !iter.is_empty() {
            self.root.insert(Root::new(*self.alloc)).bulk_push(iter, &mut self.length, *self.alloc);
        }
    }
}

impl<'a, K: 'a, V: 'a> NodeRef<marker::Immut<'a>, K, V, marker::LeafOrInternal> {
    fn assert_min_len(self, min_len: usize) {
        assert!(self.len() >= min_len, "node len {} < {}", self.len(), min_len);
        if let node::ForceResult::Internal(node) = self.force() {
            for idx in 0..=node.len() {
                let edge = unsafe { Handle::new_edge(node, idx) };
                edge.descend().assert_min_len(MIN_LEN);
            }
        }
    }
}

// Tests our value of MIN_INSERTS_HEIGHT_2. Failure may mean you just need to
// adapt that value to match a change in node::CAPACITY or the choices made
// during insertion, otherwise other test cases may fail or be less useful.
#[test]
fn test_levels() {
    let mut map = BTreeMap::new();
    map.check();
    assert_eq!(map.height(), None);
    assert_eq!(map.len(), 0);

    map.insert(0, ());
    while map.height() == Some(0) {
        let last_key = *map.last_key_value().unwrap().0;
        map.insert(last_key + 1, ());
    }
    map.check();
    // Structure:
    // - 1 element in internal root node with 2 children
    // - 6 elements in left leaf child
    // - 5 elements in right leaf child
    assert_eq!(map.height(), Some(1));
    assert_eq!(map.len(), MIN_INSERTS_HEIGHT_1, "{}", map.dump_keys());

    while map.height() == Some(1) {
        let last_key = *map.last_key_value().unwrap().0;
        map.insert(last_key + 1, ());
    }
    map.check();
    // Structure:
    // - 1 element in internal root node with 2 children
    // - 6 elements in left internal child with 7 grandchildren
    // - 42 elements in left child's 7 grandchildren with 6 elements each
    // - 5 elements in right internal child with 6 grandchildren
    // - 30 elements in right child's 5 first grandchildren with 6 elements each
    // - 5 elements in right child's last grandchild
    assert_eq!(map.height(), Some(2));
    assert_eq!(map.len(), MIN_INSERTS_HEIGHT_2, "{}", map.dump_keys());
}

// Ensures the testing infrastructure usually notices order violations.
#[test]
#[should_panic]
fn test_check_ord_chaos() {
    let gov = Governor::new();
    let map = BTreeMap::from([(Governed(1, &gov), ()), (Governed(2, &gov), ())]);
    gov.flip();
    map.check();
}

// Ensures the testing infrastructure doesn't always mind order violations.
#[test]
fn test_check_invariants_ord_chaos() {
    let gov = Governor::new();
    let map = BTreeMap::from([(Governed(1, &gov), ()), (Governed(2, &gov), ())]);
    gov.flip();
    map.check_invariants();
}

#[test]
fn test_basic_large() {
    let mut map = BTreeMap::new();
    // Miri is too slow
    let size = if cfg!(miri) { MIN_INSERTS_HEIGHT_2 } else { 10000 };
    let size = size + (size % 2); // round up to even number
    assert_eq!(map.len(), 0);

    for i in 0..size {
        assert_eq!(map.insert(i, 10 * i), None);
        assert_eq!(map.len(), i + 1);
    }

    assert_eq!(map.first_key_value(), Some((&0, &0)));
    assert_eq!(map.last_key_value(), Some((&(size - 1), &(10 * (size - 1)))));
    assert_eq!(map.first_entry().unwrap().key(), &0);
    assert_eq!(map.last_entry().unwrap().key(), &(size - 1));

    for i in 0..size {
        assert_eq!(map.get(&i).unwrap(), &(i * 10));
    }

    for i in size..size * 2 {
        assert_eq!(map.get(&i), None);
    }

    for i in 0..size {
        assert_eq!(map.insert(i, 100 * i), Some(10 * i));
        assert_eq!(map.len(), size);
    }

    for i in 0..size {
        assert_eq!(map.get(&i).unwrap(), &(i * 100));
    }

    for i in 0..size / 2 {
        assert_eq!(map.remove(&(i * 2)), Some(i * 200));
        assert_eq!(map.len(), size - i - 1);
    }

    for i in 0..size / 2 {
        assert_eq!(map.get(&(2 * i)), None);
        assert_eq!(map.get(&(2 * i + 1)).unwrap(), &(i * 200 + 100));
    }

    for i in 0..size / 2 {
        assert_eq!(map.remove(&(2 * i)), None);
        assert_eq!(map.remove(&(2 * i + 1)), Some(i * 200 + 100));
        assert_eq!(map.len(), size / 2 - i - 1);
    }
    map.check();
}

#[test]
fn test_basic_small() {
    let mut map = BTreeMap::new();
    // Empty, root is absent (None):
    assert_eq!(map.remove(&1), None);
    assert_eq!(map.len(), 0);
    assert_eq!(map.get(&1), None);
    assert_eq!(map.get_mut(&1), None);
    assert_eq!(map.first_key_value(), None);
    assert_eq!(map.last_key_value(), None);
    assert_eq!(map.keys().count(), 0);
    assert_eq!(map.values().count(), 0);
    assert_eq!(map.range(..).next(), None);
    assert_eq!(map.range(..1).next(), None);
    assert_eq!(map.range(1..).next(), None);
    assert_eq!(map.range(1..=1).next(), None);
    assert_eq!(map.range(1..2).next(), None);
    assert_eq!(map.height(), None);
    assert_eq!(map.insert(1, 1), None);
    assert_eq!(map.height(), Some(0));
    map.check();

    // 1 key-value pair:
    assert_eq!(map.len(), 1);
    assert_eq!(map.get(&1), Some(&1));
    assert_eq!(map.get_mut(&1), Some(&mut 1));
    assert_eq!(map.first_key_value(), Some((&1, &1)));
    assert_eq!(map.last_key_value(), Some((&1, &1)));
    assert_eq!(map.keys().collect::<Vec<_>>(), vec![&1]);
    assert_eq!(map.values().collect::<Vec<_>>(), vec![&1]);
    assert_eq!(map.insert(1, 2), Some(1));
    assert_eq!(map.len(), 1);
    assert_eq!(map.get(&1), Some(&2));
    assert_eq!(map.get_mut(&1), Some(&mut 2));
    assert_eq!(map.first_key_value(), Some((&1, &2)));
    assert_eq!(map.last_key_value(), Some((&1, &2)));
    assert_eq!(map.keys().collect::<Vec<_>>(), vec![&1]);
    assert_eq!(map.values().collect::<Vec<_>>(), vec![&2]);
    assert_eq!(map.insert(2, 4), None);
    assert_eq!(map.height(), Some(0));
    map.check();

    // 2 key-value pairs:
    assert_eq!(map.len(), 2);
    assert_eq!(map.get(&2), Some(&4));
    assert_eq!(map.get_mut(&2), Some(&mut 4));
    assert_eq!(map.first_key_value(), Some((&1, &2)));
    assert_eq!(map.last_key_value(), Some((&2, &4)));
    assert_eq!(map.keys().collect::<Vec<_>>(), vec![&1, &2]);
    assert_eq!(map.values().collect::<Vec<_>>(), vec![&2, &4]);
    assert_eq!(map.remove(&1), Some(2));
    assert_eq!(map.height(), Some(0));
    map.check();

    // 1 key-value pair:
    assert_eq!(map.len(), 1);
    assert_eq!(map.get(&1), None);
    assert_eq!(map.get_mut(&1), None);
    assert_eq!(map.get(&2), Some(&4));
    assert_eq!(map.get_mut(&2), Some(&mut 4));
    assert_eq!(map.first_key_value(), Some((&2, &4)));
    assert_eq!(map.last_key_value(), Some((&2, &4)));
    assert_eq!(map.keys().collect::<Vec<_>>(), vec![&2]);
    assert_eq!(map.values().collect::<Vec<_>>(), vec![&4]);
    assert_eq!(map.remove(&2), Some(4));
    assert_eq!(map.height(), Some(0));
    map.check();

    // Empty but root is owned (Some(...)):
    assert_eq!(map.len(), 0);
    assert_eq!(map.get(&1), None);
    assert_eq!(map.get_mut(&1), None);
    assert_eq!(map.first_key_value(), None);
    assert_eq!(map.last_key_value(), None);
    assert_eq!(map.keys().count(), 0);
    assert_eq!(map.values().count(), 0);
    assert_eq!(map.range(..).next(), None);
    assert_eq!(map.range(..1).next(), None);
    assert_eq!(map.range(1..).next(), None);
    assert_eq!(map.range(1..=1).next(), None);
    assert_eq!(map.range(1..2).next(), None);
    assert_eq!(map.remove(&1), None);
    assert_eq!(map.height(), Some(0));
    map.check();
}

#[test]
fn test_iter() {
    // Miri is too slow
    let size = if cfg!(miri) { 200 } else { 10000 };
    let mut map = BTreeMap::from_iter((0..size).map(|i| (i, i)));

    fn test<T>(size: usize, mut iter: T)
    where
        T: Iterator<Item = (usize, usize)>,
    {
        for i in 0..size {
            assert_eq!(iter.size_hint(), (size - i, Some(size - i)));
            assert_eq!(iter.next().unwrap(), (i, i));
        }
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
    }
    test(size, map.iter().map(|(&k, &v)| (k, v)));
    test(size, map.iter_mut().map(|(&k, &mut v)| (k, v)));
    test(size, map.into_iter());
}

#[test]
fn test_iter_rev() {
    // Miri is too slow
    let size = if cfg!(miri) { 200 } else { 10000 };
    let mut map = BTreeMap::from_iter((0..size).map(|i| (i, i)));

    fn test<T>(size: usize, mut iter: T)
    where
        T: Iterator<Item = (usize, usize)>,
    {
        for i in 0..size {
            assert_eq!(iter.size_hint(), (size - i, Some(size - i)));
            assert_eq!(iter.next().unwrap(), (size - i - 1, size - i - 1));
        }
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
    }
    test(size, map.iter().rev().map(|(&k, &v)| (k, v)));
    test(size, map.iter_mut().rev().map(|(&k, &mut v)| (k, v)));
    test(size, map.into_iter().rev());
}

// Specifically tests iter_mut's ability to mutate the value of pairs in-line.
fn do_test_iter_mut_mutation<T>(size: usize)
where
    T: Copy + Debug + Ord + TryFrom<usize>,
    <T as TryFrom<usize>>::Error: Debug,
{
    let zero = T::try_from(0).unwrap();
    let mut map = BTreeMap::from_iter((0..size).map(|i| (T::try_from(i).unwrap(), zero)));

    // Forward and backward iteration sees enough pairs (also tested elsewhere)
    assert_eq!(map.iter_mut().count(), size);
    assert_eq!(map.iter_mut().rev().count(), size);

    // Iterate forwards, trying to mutate to unique values
    for (i, (k, v)) in map.iter_mut().enumerate() {
        assert_eq!(*k, T::try_from(i).unwrap());
        assert_eq!(*v, zero);
        *v = T::try_from(i + 1).unwrap();
    }

    // Iterate backwards, checking that mutations succeeded and trying to mutate again
    for (i, (k, v)) in map.iter_mut().rev().enumerate() {
        assert_eq!(*k, T::try_from(size - i - 1).unwrap());
        assert_eq!(*v, T::try_from(size - i).unwrap());
        *v = T::try_from(2 * size - i).unwrap();
    }

    // Check that backward mutations succeeded
    for (i, (k, v)) in map.iter_mut().enumerate() {
        assert_eq!(*k, T::try_from(i).unwrap());
        assert_eq!(*v, T::try_from(size + i + 1).unwrap());
    }
    map.check();
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
#[repr(align(32))]
struct Align32(usize);

impl TryFrom<usize> for Align32 {
    type Error = ();

    fn try_from(s: usize) -> Result<Align32, ()> {
        Ok(Align32(s))
    }
}

#[test]
fn test_iter_mut_mutation() {
    // Check many alignments and trees with roots at various heights.
    do_test_iter_mut_mutation::<u8>(0);
    do_test_iter_mut_mutation::<u8>(1);
    do_test_iter_mut_mutation::<u8>(MIN_INSERTS_HEIGHT_1);
    do_test_iter_mut_mutation::<u8>(MIN_INSERTS_HEIGHT_2);
    do_test_iter_mut_mutation::<u16>(1);
    do_test_iter_mut_mutation::<u16>(MIN_INSERTS_HEIGHT_1);
    do_test_iter_mut_mutation::<u16>(MIN_INSERTS_HEIGHT_2);
    do_test_iter_mut_mutation::<u32>(1);
    do_test_iter_mut_mutation::<u32>(MIN_INSERTS_HEIGHT_1);
    do_test_iter_mut_mutation::<u32>(MIN_INSERTS_HEIGHT_2);
    do_test_iter_mut_mutation::<u64>(1);
    do_test_iter_mut_mutation::<u64>(MIN_INSERTS_HEIGHT_1);
    do_test_iter_mut_mutation::<u64>(MIN_INSERTS_HEIGHT_2);
    do_test_iter_mut_mutation::<u128>(1);
    do_test_iter_mut_mutation::<u128>(MIN_INSERTS_HEIGHT_1);
    do_test_iter_mut_mutation::<u128>(MIN_INSERTS_HEIGHT_2);
    do_test_iter_mut_mutation::<Align32>(1);
    do_test_iter_mut_mutation::<Align32>(MIN_INSERTS_HEIGHT_1);
    do_test_iter_mut_mutation::<Align32>(MIN_INSERTS_HEIGHT_2);
}

#[test]
fn test_values_mut() {
    let mut a = BTreeMap::from_iter((0..MIN_INSERTS_HEIGHT_2).map(|i| (i, i)));
    test_all_refs(&mut 13, a.values_mut());
    a.check();
}

#[test]
fn test_values_mut_mutation() {
    let mut a = BTreeMap::new();
    a.insert(1, String::from("hello"));
    a.insert(2, String::from("goodbye"));

    for value in a.values_mut() {
        value.push_str("!");
    }

    let values = Vec::from_iter(a.values().cloned());
    assert_eq!(values, [String::from("hello!"), String::from("goodbye!")]);
    a.check();
}

#[test]
fn test_iter_entering_root_twice() {
    let mut map = BTreeMap::from([(0, 0), (1, 1)]);
    let mut it = map.iter_mut();
    let front = it.next().unwrap();
    let back = it.next_back().unwrap();
    assert_eq!(front, (&0, &mut 0));
    assert_eq!(back, (&1, &mut 1));
    *front.1 = 24;
    *back.1 = 42;
    assert_eq!(front, (&0, &mut 24));
    assert_eq!(back, (&1, &mut 42));
    assert_eq!(it.next(), None);
    assert_eq!(it.next_back(), None);
    map.check();
}

#[test]
fn test_iter_descending_to_same_node_twice() {
    let mut map = BTreeMap::from_iter((0..MIN_INSERTS_HEIGHT_1).map(|i| (i, i)));
    let mut it = map.iter_mut();
    // Descend into first child.
    let front = it.next().unwrap();
    // Descend into first child again, after running through second child.
    while it.next_back().is_some() {}
    // Check immutable access.
    assert_eq!(front, (&0, &mut 0));
    // Perform mutable access.
    *front.1 = 42;
    map.check();
}

#[test]
fn test_iter_mixed() {
    // Miri is too slow
    let size = if cfg!(miri) { 200 } else { 10000 };

    let mut map = BTreeMap::from_iter((0..size).map(|i| (i, i)));

    fn test<T>(size: usize, mut iter: T)
    where
        T: Iterator<Item = (usize, usize)> + DoubleEndedIterator,
    {
        for i in 0..size / 4 {
            assert_eq!(iter.size_hint(), (size - i * 2, Some(size - i * 2)));
            assert_eq!(iter.next().unwrap(), (i, i));
            assert_eq!(iter.next_back().unwrap(), (size - i - 1, size - i - 1));
        }
        for i in size / 4..size * 3 / 4 {
            assert_eq!(iter.size_hint(), (size * 3 / 4 - i, Some(size * 3 / 4 - i)));
            assert_eq!(iter.next().unwrap(), (i, i));
        }
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
    }
    test(size, map.iter().map(|(&k, &v)| (k, v)));
    test(size, map.iter_mut().map(|(&k, &mut v)| (k, v)));
    test(size, map.into_iter());
}

#[test]
fn test_iter_min_max() {
    let mut a = BTreeMap::new();
    assert_eq!(a.iter().min(), None);
    assert_eq!(a.iter().max(), None);
    assert_eq!(a.iter_mut().min(), None);
    assert_eq!(a.iter_mut().max(), None);
    assert_eq!(a.range(..).min(), None);
    assert_eq!(a.range(..).max(), None);
    assert_eq!(a.range_mut(..).min(), None);
    assert_eq!(a.range_mut(..).max(), None);
    assert_eq!(a.keys().min(), None);
    assert_eq!(a.keys().max(), None);
    assert_eq!(a.values().min(), None);
    assert_eq!(a.values().max(), None);
    assert_eq!(a.values_mut().min(), None);
    assert_eq!(a.values_mut().max(), None);
    a.insert(1, 42);
    a.insert(2, 24);
    assert_eq!(a.iter().min(), Some((&1, &42)));
    assert_eq!(a.iter().max(), Some((&2, &24)));
    assert_eq!(a.iter_mut().min(), Some((&1, &mut 42)));
    assert_eq!(a.iter_mut().max(), Some((&2, &mut 24)));
    assert_eq!(a.range(..).min(), Some((&1, &42)));
    assert_eq!(a.range(..).max(), Some((&2, &24)));
    assert_eq!(a.range_mut(..).min(), Some((&1, &mut 42)));
    assert_eq!(a.range_mut(..).max(), Some((&2, &mut 24)));
    assert_eq!(a.keys().min(), Some(&1));
    assert_eq!(a.keys().max(), Some(&2));
    assert_eq!(a.values().min(), Some(&24));
    assert_eq!(a.values().max(), Some(&42));
    assert_eq!(a.values_mut().min(), Some(&mut 24));
    assert_eq!(a.values_mut().max(), Some(&mut 42));
    a.check();
}

fn range_keys(map: &BTreeMap<i32, i32>, range: impl RangeBounds<i32>) -> Vec<i32> {
    Vec::from_iter(map.range(range).map(|(&k, &v)| {
        assert_eq!(k, v);
        k
    }))
}

#[test]
fn test_range_small() {
    let size = 4;

    let all = Vec::from_iter(1..=size);
    let (first, last) = (vec![all[0]], vec![all[size as usize - 1]]);
    let map = BTreeMap::from_iter(all.iter().copied().map(|i| (i, i)));

    assert_eq!(range_keys(&map, (Excluded(0), Excluded(size + 1))), all);
    assert_eq!(range_keys(&map, (Excluded(0), Included(size + 1))), all);
    assert_eq!(range_keys(&map, (Excluded(0), Included(size))), all);
    assert_eq!(range_keys(&map, (Excluded(0), Unbounded)), all);
    assert_eq!(range_keys(&map, (Included(0), Excluded(size + 1))), all);
    assert_eq!(range_keys(&map, (Included(0), Included(size + 1))), all);
    assert_eq!(range_keys(&map, (Included(0), Included(size))), all);
    assert_eq!(range_keys(&map, (Included(0), Unbounded)), all);
    assert_eq!(range_keys(&map, (Included(1), Excluded(size + 1))), all);
    assert_eq!(range_keys(&map, (Included(1), Included(size + 1))), all);
    assert_eq!(range_keys(&map, (Included(1), Included(size))), all);
    assert_eq!(range_keys(&map, (Included(1), Unbounded)), all);
    assert_eq!(range_keys(&map, (Unbounded, Excluded(size + 1))), all);
    assert_eq!(range_keys(&map, (Unbounded, Included(size + 1))), all);
    assert_eq!(range_keys(&map, (Unbounded, Included(size))), all);
    assert_eq!(range_keys(&map, ..), all);

    assert_eq!(range_keys(&map, (Excluded(0), Excluded(1))), vec![]);
    assert_eq!(range_keys(&map, (Excluded(0), Included(0))), vec![]);
    assert_eq!(range_keys(&map, (Included(0), Included(0))), vec![]);
    assert_eq!(range_keys(&map, (Included(0), Excluded(1))), vec![]);
    assert_eq!(range_keys(&map, (Unbounded, Excluded(1))), vec![]);
    assert_eq!(range_keys(&map, (Unbounded, Included(0))), vec![]);
    assert_eq!(range_keys(&map, (Excluded(0), Excluded(2))), first);
    assert_eq!(range_keys(&map, (Excluded(0), Included(1))), first);
    assert_eq!(range_keys(&map, (Included(0), Excluded(2))), first);
    assert_eq!(range_keys(&map, (Included(0), Included(1))), first);
    assert_eq!(range_keys(&map, (Included(1), Excluded(2))), first);
    assert_eq!(range_keys(&map, (Included(1), Included(1))), first);
    assert_eq!(range_keys(&map, (Unbounded, Excluded(2))), first);
    assert_eq!(range_keys(&map, (Unbounded, Included(1))), first);
    assert_eq!(range_keys(&map, (Excluded(size - 1), Excluded(size + 1))), last);
    assert_eq!(range_keys(&map, (Excluded(size - 1), Included(size + 1))), last);
    assert_eq!(range_keys(&map, (Excluded(size - 1), Included(size))), last);
    assert_eq!(range_keys(&map, (Excluded(size - 1), Unbounded)), last);
    assert_eq!(range_keys(&map, (Included(size), Excluded(size + 1))), last);
    assert_eq!(range_keys(&map, (Included(size), Included(size + 1))), last);
    assert_eq!(range_keys(&map, (Included(size), Included(size))), last);
    assert_eq!(range_keys(&map, (Included(size), Unbounded)), last);
    assert_eq!(range_keys(&map, (Excluded(size), Excluded(size + 1))), vec![]);
    assert_eq!(range_keys(&map, (Excluded(size), Included(size))), vec![]);
    assert_eq!(range_keys(&map, (Excluded(size), Unbounded)), vec![]);
    assert_eq!(range_keys(&map, (Included(size + 1), Excluded(size + 1))), vec![]);
    assert_eq!(range_keys(&map, (Included(size + 1), Included(size + 1))), vec![]);
    assert_eq!(range_keys(&map, (Included(size + 1), Unbounded)), vec![]);

    assert_eq!(range_keys(&map, ..3), vec![1, 2]);
    assert_eq!(range_keys(&map, 3..), vec![3, 4]);
    assert_eq!(range_keys(&map, 2..=3), vec![2, 3]);
}

#[test]
fn test_range_height_1() {
    // Tests tree with a root and 2 leaves. We test around the middle of the
    // keys because one of those is the single key in the root node.
    let map = BTreeMap::from_iter((0..MIN_INSERTS_HEIGHT_1 as i32).map(|i| (i, i)));
    let middle = MIN_INSERTS_HEIGHT_1 as i32 / 2;
    for root in middle - 2..=middle + 2 {
        assert_eq!(range_keys(&map, (Excluded(root), Excluded(root + 1))), vec![]);
        assert_eq!(range_keys(&map, (Excluded(root), Included(root + 1))), vec![root + 1]);
        assert_eq!(range_keys(&map, (Included(root), Excluded(root + 1))), vec![root]);
        assert_eq!(range_keys(&map, (Included(root), Included(root + 1))), vec![root, root + 1]);

        assert_eq!(range_keys(&map, (Excluded(root - 1), Excluded(root))), vec![]);
        assert_eq!(range_keys(&map, (Included(root - 1), Excluded(root))), vec![root - 1]);
        assert_eq!(range_keys(&map, (Excluded(root - 1), Included(root))), vec![root]);
        assert_eq!(range_keys(&map, (Included(root - 1), Included(root))), vec![root - 1, root]);
    }
}

#[test]
fn test_range_large() {
    let size = 200;

    let all = Vec::from_iter(1..=size);
    let (first, last) = (vec![all[0]], vec![all[size as usize - 1]]);
    let map = BTreeMap::from_iter(all.iter().copied().map(|i| (i, i)));

    assert_eq!(range_keys(&map, (Excluded(0), Excluded(size + 1))), all);
    assert_eq!(range_keys(&map, (Excluded(0), Included(size + 1))), all);
    assert_eq!(range_keys(&map, (Excluded(0), Included(size))), all);
    assert_eq!(range_keys(&map, (Excluded(0), Unbounded)), all);
    assert_eq!(range_keys(&map, (Included(0), Excluded(size + 1))), all);
    assert_eq!(range_keys(&map, (Included(0), Included(size + 1))), all);
    assert_eq!(range_keys(&map, (Included(0), Included(size))), all);
    assert_eq!(range_keys(&map, (Included(0), Unbounded)), all);
    assert_eq!(range_keys(&map, (Included(1), Excluded(size + 1))), all);
    assert_eq!(range_keys(&map, (Included(1), Included(size + 1))), all);
    assert_eq!(range_keys(&map, (Included(1), Included(size))), all);
    assert_eq!(range_keys(&map, (Included(1), Unbounded)), all);
    assert_eq!(range_keys(&map, (Unbounded, Excluded(size + 1))), all);
    assert_eq!(range_keys(&map, (Unbounded, Included(size + 1))), all);
    assert_eq!(range_keys(&map, (Unbounded, Included(size))), all);
    assert_eq!(range_keys(&map, ..), all);

    assert_eq!(range_keys(&map, (Excluded(0), Excluded(1))), vec![]);
    assert_eq!(range_keys(&map, (Excluded(0), Included(0))), vec![]);
    assert_eq!(range_keys(&map, (Included(0), Included(0))), vec![]);
    assert_eq!(range_keys(&map, (Included(0), Excluded(1))), vec![]);
    assert_eq!(range_keys(&map, (Unbounded, Excluded(1))), vec![]);
    assert_eq!(range_keys(&map, (Unbounded, Included(0))), vec![]);
    assert_eq!(range_keys(&map, (Excluded(0), Excluded(2))), first);
    assert_eq!(range_keys(&map, (Excluded(0), Included(1))), first);
    assert_eq!(range_keys(&map, (Included(0), Excluded(2))), first);
    assert_eq!(range_keys(&map, (Included(0), Included(1))), first);
    assert_eq!(range_keys(&map, (Included(1), Excluded(2))), first);
    assert_eq!(range_keys(&map, (Included(1), Included(1))), first);
    assert_eq!(range_keys(&map, (Unbounded, Excluded(2))), first);
    assert_eq!(range_keys(&map, (Unbounded, Included(1))), first);
    assert_eq!(range_keys(&map, (Excluded(size - 1), Excluded(size + 1))), last);
    assert_eq!(range_keys(&map, (Excluded(size - 1), Included(size + 1))), last);
    assert_eq!(range_keys(&map, (Excluded(size - 1), Included(size))), last);
    assert_eq!(range_keys(&map, (Excluded(size - 1), Unbounded)), last);
    assert_eq!(range_keys(&map, (Included(size), Excluded(size + 1))), last);
    assert_eq!(range_keys(&map, (Included(size), Included(size + 1))), last);
    assert_eq!(range_keys(&map, (Included(size), Included(size))), last);
    assert_eq!(range_keys(&map, (Included(size), Unbounded)), last);
    assert_eq!(range_keys(&map, (Excluded(size), Excluded(size + 1))), vec![]);
    assert_eq!(range_keys(&map, (Excluded(size), Included(size))), vec![]);
    assert_eq!(range_keys(&map, (Excluded(size), Unbounded)), vec![]);
    assert_eq!(range_keys(&map, (Included(size + 1), Excluded(size + 1))), vec![]);
    assert_eq!(range_keys(&map, (Included(size + 1), Included(size + 1))), vec![]);
    assert_eq!(range_keys(&map, (Included(size + 1), Unbounded)), vec![]);

    fn check<'a, L, R>(lhs: L, rhs: R)
    where
        L: IntoIterator<Item = (&'a i32, &'a i32)>,
        R: IntoIterator<Item = (&'a i32, &'a i32)>,
    {
        assert_eq!(Vec::from_iter(lhs), Vec::from_iter(rhs));
    }

    check(map.range(..=100), map.range(..101));
    check(map.range(5..=8), vec![(&5, &5), (&6, &6), (&7, &7), (&8, &8)]);
    check(map.range(-1..=2), vec![(&1, &1), (&2, &2)]);
}

#[test]
fn test_range_inclusive_max_value() {
    let max = usize::MAX;
    let map = BTreeMap::from([(max, 0)]);
    assert_eq!(Vec::from_iter(map.range(max..=max)), &[(&max, &0)]);
}

#[test]
fn test_range_equal_empty_cases() {
    let map = BTreeMap::from_iter((0..5).map(|i| (i, i)));
    assert_eq!(map.range((Included(2), Excluded(2))).next(), None);
    assert_eq!(map.range((Excluded(2), Included(2))).next(), None);
}

#[test]
#[should_panic]
fn test_range_equal_excluded() {
    let map = BTreeMap::from_iter((0..5).map(|i| (i, i)));
    let _ = map.range((Excluded(2), Excluded(2)));
}

#[test]
#[should_panic]
fn test_range_backwards_1() {
    let map = BTreeMap::from_iter((0..5).map(|i| (i, i)));
    let _ = map.range((Included(3), Included(2)));
}

#[test]
#[should_panic]
fn test_range_backwards_2() {
    let map = BTreeMap::from_iter((0..5).map(|i| (i, i)));
    let _ = map.range((Included(3), Excluded(2)));
}

#[test]
#[should_panic]
fn test_range_backwards_3() {
    let map = BTreeMap::from_iter((0..5).map(|i| (i, i)));
    let _ = map.range((Excluded(3), Included(2)));
}

#[test]
#[should_panic]
fn test_range_backwards_4() {
    let map = BTreeMap::from_iter((0..5).map(|i| (i, i)));
    let _ = map.range((Excluded(3), Excluded(2)));
}

#[test]
fn test_range_finding_ill_order_in_map() {
    let mut map = BTreeMap::new();
    map.insert(Cyclic3::B, ());
    // Lacking static_assert, call `range` conditionally, to emphasise that
    // we cause a different panic than `test_range_backwards_1` does.
    // A more refined `should_panic` would be welcome.
    if Cyclic3::C < Cyclic3::A {
        let _ = map.range(Cyclic3::C..=Cyclic3::A);
    }
}

#[test]
fn test_range_finding_ill_order_in_range_ord() {
    // Has proper order the first time asked, then flips around.
    struct EvilTwin(i32);

    impl PartialOrd for EvilTwin {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    static COMPARES: AtomicUsize = AtomicUsize::new(0);
    impl Ord for EvilTwin {
        fn cmp(&self, other: &Self) -> Ordering {
            let ord = self.0.cmp(&other.0);
            if COMPARES.fetch_add(1, SeqCst) > 0 { ord.reverse() } else { ord }
        }
    }

    impl PartialEq for EvilTwin {
        fn eq(&self, other: &Self) -> bool {
            self.0.eq(&other.0)
        }
    }

    impl Eq for EvilTwin {}

    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    struct CompositeKey(i32, EvilTwin);

    impl Borrow<EvilTwin> for CompositeKey {
        fn borrow(&self) -> &EvilTwin {
            &self.1
        }
    }

    let map = BTreeMap::from_iter((0..12).map(|i| (CompositeKey(i, EvilTwin(i)), ())));
    let _ = map.range(EvilTwin(5)..=EvilTwin(7));
}

#[test]
fn test_range_1000() {
    // Miri is too slow
    let size = if cfg!(miri) { MIN_INSERTS_HEIGHT_2 as u32 } else { 1000 };
    let map = BTreeMap::from_iter((0..size).map(|i| (i, i)));

    fn test(map: &BTreeMap<u32, u32>, size: u32, min: Bound<&u32>, max: Bound<&u32>) {
        let mut kvs = map.range((min, max)).map(|(&k, &v)| (k, v));
        let mut pairs = (0..size).map(|i| (i, i));

        for (kv, pair) in kvs.by_ref().zip(pairs.by_ref()) {
            assert_eq!(kv, pair);
        }
        assert_eq!(kvs.next(), None);
        assert_eq!(pairs.next(), None);
    }
    test(&map, size, Included(&0), Excluded(&size));
    test(&map, size, Unbounded, Excluded(&size));
    test(&map, size, Included(&0), Included(&(size - 1)));
    test(&map, size, Unbounded, Included(&(size - 1)));
    test(&map, size, Included(&0), Unbounded);
    test(&map, size, Unbounded, Unbounded);
}

#[test]
fn test_range_borrowed_key() {
    let mut map = BTreeMap::new();
    map.insert("aardvark".to_string(), 1);
    map.insert("baboon".to_string(), 2);
    map.insert("coyote".to_string(), 3);
    map.insert("dingo".to_string(), 4);
    // NOTE: would like to use simply "b".."d" here...
    let mut iter = map.range::<str, _>((Included("b"), Excluded("d")));
    assert_eq!(iter.next(), Some((&"baboon".to_string(), &2)));
    assert_eq!(iter.next(), Some((&"coyote".to_string(), &3)));
    assert_eq!(iter.next(), None);
}

#[test]
fn test_range() {
    let size = 200;
    // Miri is too slow
    let step = if cfg!(miri) { 66 } else { 1 };
    let map = BTreeMap::from_iter((0..size).map(|i| (i, i)));

    for i in (0..size).step_by(step) {
        for j in (i..size).step_by(step) {
            let mut kvs = map.range((Included(&i), Included(&j))).map(|(&k, &v)| (k, v));
            let mut pairs = (i..=j).map(|i| (i, i));

            for (kv, pair) in kvs.by_ref().zip(pairs.by_ref()) {
                assert_eq!(kv, pair);
            }
            assert_eq!(kvs.next(), None);
            assert_eq!(pairs.next(), None);
        }
    }
}

#[test]
fn test_range_mut() {
    let size = 200;
    // Miri is too slow
    let step = if cfg!(miri) { 66 } else { 1 };
    let mut map = BTreeMap::from_iter((0..size).map(|i| (i, i)));

    for i in (0..size).step_by(step) {
        for j in (i..size).step_by(step) {
            let mut kvs = map.range_mut((Included(&i), Included(&j))).map(|(&k, &mut v)| (k, v));
            let mut pairs = (i..=j).map(|i| (i, i));

            for (kv, pair) in kvs.by_ref().zip(pairs.by_ref()) {
                assert_eq!(kv, pair);
            }
            assert_eq!(kvs.next(), None);
            assert_eq!(pairs.next(), None);
        }
    }
    map.check();
}

#[should_panic(expected = "range start is greater than range end in BTreeMap")]
#[test]
fn test_range_panic_1() {
    let mut map = BTreeMap::new();
    map.insert(3, "a");
    map.insert(5, "b");
    map.insert(8, "c");

    let _invalid_range = map.range((Included(&8), Included(&3)));
}

#[should_panic(expected = "range start and end are equal and excluded in BTreeMap")]
#[test]
fn test_range_panic_2() {
    let mut map = BTreeMap::new();
    map.insert(3, "a");
    map.insert(5, "b");
    map.insert(8, "c");

    let _invalid_range = map.range((Excluded(&5), Excluded(&5)));
}

#[should_panic(expected = "range start and end are equal and excluded in BTreeMap")]
#[test]
fn test_range_panic_3() {
    let mut map: BTreeMap<i32, ()> = BTreeMap::new();
    map.insert(3, ());
    map.insert(5, ());
    map.insert(8, ());

    let _invalid_range = map.range((Excluded(&5), Excluded(&5)));
}

#[test]
fn test_retain() {
    let mut map = BTreeMap::from_iter((0..100).map(|x| (x, x * 10)));

    map.retain(|&k, _| k % 2 == 0);
    assert_eq!(map.len(), 50);
    assert_eq!(map[&2], 20);
    assert_eq!(map[&4], 40);
    assert_eq!(map[&6], 60);
}

mod test_extract_if {
    use super::*;

    #[test]
    fn empty() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new();
        map.extract_if(.., |_, _| unreachable!("there's nothing to decide on")).for_each(drop);
        assert_eq!(map.height(), None);
        map.check();
    }

    // Explicitly consumes the iterator, where most test cases drop it instantly.
    #[test]
    fn consumed_keeping_all() {
        let pairs = (0..3).map(|i| (i, i));
        let mut map = BTreeMap::from_iter(pairs);
        assert!(map.extract_if(.., |_, _| false).eq(iter::empty()));
        map.check();
    }

    // Explicitly consumes the iterator, where most test cases drop it instantly.
    #[test]
    fn consumed_removing_all() {
        let pairs = (0..3).map(|i| (i, i));
        let mut map = BTreeMap::from_iter(pairs.clone());
        assert!(map.extract_if(.., |_, _| true).eq(pairs));
        assert!(map.is_empty());
        map.check();
    }

    #[test]
    fn consumed_removing_some() {
        let pairs = (0..3).map(|i| (i, i));
        let map = BTreeMap::from_iter(pairs);
        for x in 0..3 {
            for y in 0..3 {
                let mut map = map.clone();
                assert!(map.extract_if(x..y, |_, _| true).eq((x..y).map(|i| (i, i))));
                for i in 0..3 {
                    assert_ne!(map.contains_key(&i), (x..y).contains(&i));
                }
            }
        }
        for x in 0..3 {
            for y in 0..2 {
                let mut map = map.clone();
                assert!(map.extract_if(x..=y, |_, _| true).eq((x..=y).map(|i| (i, i))));
                for i in 0..3 {
                    assert_ne!(map.contains_key(&i), (x..=y).contains(&i));
                }
            }
        }
    }

    // Explicitly consumes the iterator and modifies values through it.
    #[test]
    fn mutating_and_keeping() {
        let pairs = (0..3).map(|i| (i, i));
        let mut map = BTreeMap::from_iter(pairs);
        assert!(
            map.extract_if(.., |_, v| {
                *v += 6;
                false
            })
            .eq(iter::empty())
        );
        assert!(map.keys().copied().eq(0..3));
        assert!(map.values().copied().eq(6..9));
        map.check();
    }

    // Explicitly consumes the iterator and modifies values through it.
    #[test]
    fn mutating_and_removing() {
        let pairs = (0..3).map(|i| (i, i));
        let mut map = BTreeMap::from_iter(pairs);
        assert!(
            map.extract_if(.., |_, v| {
                *v += 6;
                true
            })
            .eq((0..3).map(|i| (i, i + 6)))
        );
        assert!(map.is_empty());
        map.check();
    }

    #[test]
    fn underfull_keeping_all() {
        let pairs = (0..3).map(|i| (i, i));
        let mut map = BTreeMap::from_iter(pairs);
        map.extract_if(.., |_, _| false).for_each(drop);
        assert!(map.keys().copied().eq(0..3));
        map.check();
    }

    #[test]
    fn underfull_removing_one() {
        let pairs = (0..3).map(|i| (i, i));
        for doomed in 0..3 {
            let mut map = BTreeMap::from_iter(pairs.clone());
            map.extract_if(.., |i, _| *i == doomed).for_each(drop);
            assert_eq!(map.len(), 2);
            map.check();
        }
    }

    #[test]
    fn underfull_keeping_one() {
        let pairs = (0..3).map(|i| (i, i));
        for sacred in 0..3 {
            let mut map = BTreeMap::from_iter(pairs.clone());
            map.extract_if(.., |i, _| *i != sacred).for_each(drop);
            assert!(map.keys().copied().eq(sacred..=sacred));
            map.check();
        }
    }

    #[test]
    fn underfull_removing_all() {
        let pairs = (0..3).map(|i| (i, i));
        let mut map = BTreeMap::from_iter(pairs);
        map.extract_if(.., |_, _| true).for_each(drop);
        assert!(map.is_empty());
        map.check();
    }

    #[test]
    fn height_0_keeping_all() {
        let pairs = (0..node::CAPACITY).map(|i| (i, i));
        let mut map = BTreeMap::from_iter(pairs);
        map.extract_if(.., |_, _| false).for_each(drop);
        assert!(map.keys().copied().eq(0..node::CAPACITY));
        map.check();
    }

    #[test]
    fn height_0_removing_one() {
        let pairs = (0..node::CAPACITY).map(|i| (i, i));
        for doomed in 0..node::CAPACITY {
            let mut map = BTreeMap::from_iter(pairs.clone());
            map.extract_if(.., |i, _| *i == doomed).for_each(drop);
            assert_eq!(map.len(), node::CAPACITY - 1);
            map.check();
        }
    }

    #[test]
    fn height_0_keeping_one() {
        let pairs = (0..node::CAPACITY).map(|i| (i, i));
        for sacred in 0..node::CAPACITY {
            let mut map = BTreeMap::from_iter(pairs.clone());
            map.extract_if(.., |i, _| *i != sacred).for_each(drop);
            assert!(map.keys().copied().eq(sacred..=sacred));
            map.check();
        }
    }

    #[test]
    fn height_0_removing_all() {
        let pairs = (0..node::CAPACITY).map(|i| (i, i));
        let mut map = BTreeMap::from_iter(pairs);
        map.extract_if(.., |_, _| true).for_each(drop);
        assert!(map.is_empty());
        map.check();
    }

    #[test]
    fn height_0_keeping_half() {
        let mut map = BTreeMap::from_iter((0..16).map(|i| (i, i)));
        assert_eq!(map.extract_if(.., |i, _| *i % 2 == 0).count(), 8);
        assert_eq!(map.len(), 8);
        map.check();
    }

    #[test]
    fn height_1_removing_all() {
        let pairs = (0..MIN_INSERTS_HEIGHT_1).map(|i| (i, i));
        let mut map = BTreeMap::from_iter(pairs);
        map.extract_if(.., |_, _| true).for_each(drop);
        assert!(map.is_empty());
        map.check();
    }

    #[test]
    fn height_1_removing_one() {
        let pairs = (0..MIN_INSERTS_HEIGHT_1).map(|i| (i, i));
        for doomed in 0..MIN_INSERTS_HEIGHT_1 {
            let mut map = BTreeMap::from_iter(pairs.clone());
            map.extract_if(.., |i, _| *i == doomed).for_each(drop);
            assert_eq!(map.len(), MIN_INSERTS_HEIGHT_1 - 1);
            map.check();
        }
    }

    #[test]
    fn height_1_keeping_one() {
        let pairs = (0..MIN_INSERTS_HEIGHT_1).map(|i| (i, i));
        for sacred in 0..MIN_INSERTS_HEIGHT_1 {
            let mut map = BTreeMap::from_iter(pairs.clone());
            map.extract_if(.., |i, _| *i != sacred).for_each(drop);
            assert!(map.keys().copied().eq(sacred..=sacred));
            map.check();
        }
    }

    #[test]
    fn height_2_removing_one() {
        let pairs = (0..MIN_INSERTS_HEIGHT_2).map(|i| (i, i));
        for doomed in (0..MIN_INSERTS_HEIGHT_2).step_by(12) {
            let mut map = BTreeMap::from_iter(pairs.clone());
            map.extract_if(.., |i, _| *i == doomed).for_each(drop);
            assert_eq!(map.len(), MIN_INSERTS_HEIGHT_2 - 1);
            map.check();
        }
    }

    #[test]
    fn height_2_keeping_one() {
        let pairs = (0..MIN_INSERTS_HEIGHT_2).map(|i| (i, i));
        for sacred in (0..MIN_INSERTS_HEIGHT_2).step_by(12) {
            let mut map = BTreeMap::from_iter(pairs.clone());
            map.extract_if(.., |i, _| *i != sacred).for_each(drop);
            assert!(map.keys().copied().eq(sacred..=sacred));
            map.check();
        }
    }

    #[test]
    fn height_2_removing_all() {
        let pairs = (0..MIN_INSERTS_HEIGHT_2).map(|i| (i, i));
        let mut map = BTreeMap::from_iter(pairs);
        map.extract_if(.., |_, _| true).for_each(drop);
        assert!(map.is_empty());
        map.check();
    }

    #[test]
    #[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
    fn drop_panic_leak() {
        let a = CrashTestDummy::new(0);
        let b = CrashTestDummy::new(1);
        let c = CrashTestDummy::new(2);
        let mut map = BTreeMap::new();
        map.insert(a.spawn(Panic::Never), ());
        map.insert(b.spawn(Panic::InDrop), ());
        map.insert(c.spawn(Panic::Never), ());

        catch_unwind(move || map.extract_if(.., |dummy, _| dummy.query(true)).for_each(drop))
            .unwrap_err();

        assert_eq!(a.queried(), 1);
        assert_eq!(b.queried(), 1);
        assert_eq!(c.queried(), 0);
        assert_eq!(a.dropped(), 1);
        assert_eq!(b.dropped(), 1);
        assert_eq!(c.dropped(), 1);
    }

    #[test]
    #[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
    fn pred_panic_leak() {
        let a = CrashTestDummy::new(0);
        let b = CrashTestDummy::new(1);
        let c = CrashTestDummy::new(2);
        let mut map = BTreeMap::new();
        map.insert(a.spawn(Panic::Never), ());
        map.insert(b.spawn(Panic::InQuery), ());
        map.insert(c.spawn(Panic::InQuery), ());

        catch_unwind(AssertUnwindSafe(|| {
            map.extract_if(.., |dummy, _| dummy.query(true)).for_each(drop)
        }))
        .unwrap_err();

        assert_eq!(a.queried(), 1);
        assert_eq!(b.queried(), 1);
        assert_eq!(c.queried(), 0);
        assert_eq!(a.dropped(), 1);
        assert_eq!(b.dropped(), 0);
        assert_eq!(c.dropped(), 0);
        assert_eq!(map.len(), 2);
        assert_eq!(map.first_entry().unwrap().key().id(), 1);
        assert_eq!(map.last_entry().unwrap().key().id(), 2);
        map.check();
    }

    // Same as above, but attempt to use the iterator again after the panic in the predicate
    #[test]
    #[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
    fn pred_panic_reuse() {
        let a = CrashTestDummy::new(0);
        let b = CrashTestDummy::new(1);
        let c = CrashTestDummy::new(2);
        let mut map = BTreeMap::new();
        map.insert(a.spawn(Panic::Never), ());
        map.insert(b.spawn(Panic::InQuery), ());
        map.insert(c.spawn(Panic::InQuery), ());

        {
            let mut it = map.extract_if(.., |dummy, _| dummy.query(true));
            catch_unwind(AssertUnwindSafe(|| while it.next().is_some() {})).unwrap_err();
            // Iterator behavior after a panic is explicitly unspecified,
            // so this is just the current implementation:
            let result = catch_unwind(AssertUnwindSafe(|| it.next()));
            assert!(matches!(result, Ok(None)));
        }

        assert_eq!(a.queried(), 1);
        assert_eq!(b.queried(), 1);
        assert_eq!(c.queried(), 0);
        assert_eq!(a.dropped(), 1);
        assert_eq!(b.dropped(), 0);
        assert_eq!(c.dropped(), 0);
        assert_eq!(map.len(), 2);
        assert_eq!(map.first_entry().unwrap().key().id(), 1);
        assert_eq!(map.last_entry().unwrap().key().id(), 2);
        map.check();
    }
}

#[test]
fn test_borrow() {
    // make sure these compile -- using the Borrow trait
    {
        let mut map = BTreeMap::new();
        map.insert("0".to_string(), 1);
        assert_eq!(map["0"], 1);
    }

    {
        let mut map = BTreeMap::new();
        map.insert(Box::new(0), 1);
        assert_eq!(map[&0], 1);
    }

    {
        let mut map = BTreeMap::new();
        map.insert(Box::new([0, 1]) as Box<[i32]>, 1);
        assert_eq!(map[&[0, 1][..]], 1);
    }

    {
        let mut map = BTreeMap::new();
        map.insert(Rc::new(0), 1);
        assert_eq!(map[&0], 1);
    }

    #[allow(dead_code)]
    fn get<T: Ord>(v: &BTreeMap<Box<T>, ()>, t: &T) {
        let _ = v.get(t);
    }

    #[allow(dead_code)]
    fn get_mut<T: Ord>(v: &mut BTreeMap<Box<T>, ()>, t: &T) {
        let _ = v.get_mut(t);
    }

    #[allow(dead_code)]
    fn get_key_value<T: Ord>(v: &BTreeMap<Box<T>, ()>, t: &T) {
        let _ = v.get_key_value(t);
    }

    #[allow(dead_code)]
    fn contains_key<T: Ord>(v: &BTreeMap<Box<T>, ()>, t: &T) {
        let _ = v.contains_key(t);
    }

    #[allow(dead_code)]
    fn range<T: Ord>(v: &BTreeMap<Box<T>, ()>, t: T) {
        let _ = v.range(t..);
    }

    #[allow(dead_code)]
    fn range_mut<T: Ord>(v: &mut BTreeMap<Box<T>, ()>, t: T) {
        let _ = v.range_mut(t..);
    }

    #[allow(dead_code)]
    fn remove<T: Ord>(v: &mut BTreeMap<Box<T>, ()>, t: &T) {
        v.remove(t);
    }

    #[allow(dead_code)]
    fn remove_entry<T: Ord>(v: &mut BTreeMap<Box<T>, ()>, t: &T) {
        v.remove_entry(t);
    }

    #[allow(dead_code)]
    fn split_off<T: Ord>(v: &mut BTreeMap<Box<T>, ()>, t: &T) {
        v.split_off(t);
    }
}

#[test]
fn test_entry() {
    let xs = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

    let mut map = BTreeMap::from(xs);

    // Existing key (insert)
    match map.entry(1) {
        Vacant(_) => unreachable!(),
        Occupied(mut view) => {
            assert_eq!(view.get(), &10);
            assert_eq!(view.insert(100), 10);
        }
    }
    assert_eq!(map.get(&1).unwrap(), &100);
    assert_eq!(map.len(), 6);

    // Existing key (update)
    match map.entry(2) {
        Vacant(_) => unreachable!(),
        Occupied(mut view) => {
            let v = view.get_mut();
            *v *= 10;
        }
    }
    assert_eq!(map.get(&2).unwrap(), &200);
    assert_eq!(map.len(), 6);
    map.check();

    // Existing key (take)
    match map.entry(3) {
        Vacant(_) => unreachable!(),
        Occupied(view) => {
            assert_eq!(view.remove(), 30);
        }
    }
    assert_eq!(map.get(&3), None);
    assert_eq!(map.len(), 5);
    map.check();

    // Inexistent key (insert)
    match map.entry(10) {
        Occupied(_) => unreachable!(),
        Vacant(view) => {
            assert_eq!(*view.insert(1000), 1000);
        }
    }
    assert_eq!(map.get(&10).unwrap(), &1000);
    assert_eq!(map.len(), 6);
    map.check();
}

#[test]
fn test_extend_ref() {
    let mut a = BTreeMap::new();
    a.insert(1, "one");
    let mut b = BTreeMap::new();
    b.insert(2, "two");
    b.insert(3, "three");

    a.extend(&b);

    assert_eq!(a.len(), 3);
    assert_eq!(a[&1], "one");
    assert_eq!(a[&2], "two");
    assert_eq!(a[&3], "three");
    a.check();
}

#[test]
fn test_zst() {
    let mut m = BTreeMap::new();
    assert_eq!(m.len(), 0);

    assert_eq!(m.insert((), ()), None);
    assert_eq!(m.len(), 1);

    assert_eq!(m.insert((), ()), Some(()));
    assert_eq!(m.len(), 1);
    assert_eq!(m.iter().count(), 1);

    m.clear();
    assert_eq!(m.len(), 0);

    for _ in 0..100 {
        m.insert((), ());
    }

    assert_eq!(m.len(), 1);
    assert_eq!(m.iter().count(), 1);
    m.check();
}

// This test's only purpose is to ensure that zero-sized keys with nonsensical orderings
// do not cause segfaults when used with zero-sized values. All other map behavior is
// undefined.
#[test]
fn test_bad_zst() {
    #[derive(Clone, Copy, Debug)]
    struct Bad;

    impl PartialEq for Bad {
        fn eq(&self, _: &Self) -> bool {
            false
        }
    }

    impl Eq for Bad {}

    impl PartialOrd for Bad {
        fn partial_cmp(&self, _: &Self) -> Option<Ordering> {
            Some(Ordering::Less)
        }
    }

    impl Ord for Bad {
        fn cmp(&self, _: &Self) -> Ordering {
            Ordering::Less
        }
    }

    let mut m = BTreeMap::new();

    for _ in 0..100 {
        m.insert(Bad, Bad);
    }
    m.check();
}

#[test]
fn test_clear() {
    let mut map = BTreeMap::new();
    for &len in &[MIN_INSERTS_HEIGHT_1, MIN_INSERTS_HEIGHT_2, 0, node::CAPACITY] {
        for i in 0..len {
            map.insert(i, ());
        }
        assert_eq!(map.len(), len);
        map.clear();
        map.check();
        assert_eq!(map.height(), None);
    }
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_clear_drop_panic_leak() {
    let a = CrashTestDummy::new(0);
    let b = CrashTestDummy::new(1);
    let c = CrashTestDummy::new(2);

    let mut map = BTreeMap::new();
    map.insert(a.spawn(Panic::Never), ());
    map.insert(b.spawn(Panic::InDrop), ());
    map.insert(c.spawn(Panic::Never), ());

    catch_unwind(AssertUnwindSafe(|| map.clear())).unwrap_err();
    assert_eq!(a.dropped(), 1);
    assert_eq!(b.dropped(), 1);
    assert_eq!(c.dropped(), 1);
    assert_eq!(map.len(), 0);

    drop(map);
    assert_eq!(a.dropped(), 1);
    assert_eq!(b.dropped(), 1);
    assert_eq!(c.dropped(), 1);
}

#[test]
fn test_clone() {
    let mut map = BTreeMap::new();
    let size = MIN_INSERTS_HEIGHT_1;
    assert_eq!(map.len(), 0);

    for i in 0..size {
        assert_eq!(map.insert(i, 10 * i), None);
        assert_eq!(map.len(), i + 1);
        map.check();
        assert_eq!(map, map.clone());
    }

    for i in 0..size {
        assert_eq!(map.insert(i, 100 * i), Some(10 * i));
        assert_eq!(map.len(), size);
        map.check();
        assert_eq!(map, map.clone());
    }

    for i in 0..size / 2 {
        assert_eq!(map.remove(&(i * 2)), Some(i * 200));
        assert_eq!(map.len(), size - i - 1);
        map.check();
        assert_eq!(map, map.clone());
    }

    for i in 0..size / 2 {
        assert_eq!(map.remove(&(2 * i)), None);
        assert_eq!(map.remove(&(2 * i + 1)), Some(i * 200 + 100));
        assert_eq!(map.len(), size / 2 - i - 1);
        map.check();
        assert_eq!(map, map.clone());
    }

    // Test a tree with 2 semi-full levels and a tree with 3 levels.
    map = BTreeMap::from_iter((1..MIN_INSERTS_HEIGHT_2).map(|i| (i, i)));
    assert_eq!(map.len(), MIN_INSERTS_HEIGHT_2 - 1);
    assert_eq!(map, map.clone());
    map.insert(0, 0);
    assert_eq!(map.len(), MIN_INSERTS_HEIGHT_2);
    assert_eq!(map, map.clone());
    map.check();
}

fn test_clone_panic_leak(size: usize) {
    for i in 0..size {
        let dummies = Vec::from_iter((0..size).map(|id| CrashTestDummy::new(id)));
        let map = BTreeMap::from_iter(dummies.iter().map(|dummy| {
            let panic = if dummy.id == i { Panic::InClone } else { Panic::Never };
            (dummy.spawn(panic), ())
        }));

        catch_unwind(|| map.clone()).unwrap_err();
        for d in &dummies {
            assert_eq!(d.cloned(), if d.id <= i { 1 } else { 0 }, "id={}/{}", d.id, i);
            assert_eq!(d.dropped(), if d.id < i { 1 } else { 0 }, "id={}/{}", d.id, i);
        }
        assert_eq!(map.len(), size);

        drop(map);
        for d in &dummies {
            assert_eq!(d.cloned(), if d.id <= i { 1 } else { 0 }, "id={}/{}", d.id, i);
            assert_eq!(d.dropped(), if d.id < i { 2 } else { 1 }, "id={}/{}", d.id, i);
        }
    }
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_clone_panic_leak_height_0() {
    test_clone_panic_leak(3)
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_clone_panic_leak_height_1() {
    test_clone_panic_leak(MIN_INSERTS_HEIGHT_1)
}

#[test]
fn test_clone_from() {
    let mut map1 = BTreeMap::new();
    let max_size = MIN_INSERTS_HEIGHT_1;

    // Range to max_size inclusive, because i is the size of map1 being tested.
    for i in 0..=max_size {
        let mut map2 = BTreeMap::new();
        for j in 0..i {
            let mut map1_copy = map2.clone();
            map1_copy.clone_from(&map1); // small cloned from large
            assert_eq!(map1_copy, map1);
            let mut map2_copy = map1.clone();
            map2_copy.clone_from(&map2); // large cloned from small
            assert_eq!(map2_copy, map2);
            map2.insert(100 * j + 1, 2 * j + 1);
        }
        map2.clone_from(&map1); // same length
        map2.check();
        assert_eq!(map2, map1);
        map1.insert(i, 10 * i);
        map1.check();
    }
}

#[allow(dead_code)]
fn assert_covariance() {
    fn map_key<'new>(v: BTreeMap<&'static str, ()>) -> BTreeMap<&'new str, ()> {
        v
    }
    fn map_val<'new>(v: BTreeMap<(), &'static str>) -> BTreeMap<(), &'new str> {
        v
    }

    fn iter_key<'a, 'new>(v: Iter<'a, &'static str, ()>) -> Iter<'a, &'new str, ()> {
        v
    }
    fn iter_val<'a, 'new>(v: Iter<'a, (), &'static str>) -> Iter<'a, (), &'new str> {
        v
    }

    fn into_iter_key<'new>(v: IntoIter<&'static str, ()>) -> IntoIter<&'new str, ()> {
        v
    }
    fn into_iter_val<'new>(v: IntoIter<(), &'static str>) -> IntoIter<(), &'new str> {
        v
    }

    fn into_keys_key<'new>(v: IntoKeys<&'static str, ()>) -> IntoKeys<&'new str, ()> {
        v
    }
    fn into_keys_val<'new>(v: IntoKeys<(), &'static str>) -> IntoKeys<(), &'new str> {
        v
    }

    fn into_values_key<'new>(v: IntoValues<&'static str, ()>) -> IntoValues<&'new str, ()> {
        v
    }
    fn into_values_val<'new>(v: IntoValues<(), &'static str>) -> IntoValues<(), &'new str> {
        v
    }

    fn range_key<'a, 'new>(v: Range<'a, &'static str, ()>) -> Range<'a, &'new str, ()> {
        v
    }
    fn range_val<'a, 'new>(v: Range<'a, (), &'static str>) -> Range<'a, (), &'new str> {
        v
    }

    fn keys_key<'a, 'new>(v: Keys<'a, &'static str, ()>) -> Keys<'a, &'new str, ()> {
        v
    }
    fn keys_val<'a, 'new>(v: Keys<'a, (), &'static str>) -> Keys<'a, (), &'new str> {
        v
    }

    fn values_key<'a, 'new>(v: Values<'a, &'static str, ()>) -> Values<'a, &'new str, ()> {
        v
    }
    fn values_val<'a, 'new>(v: Values<'a, (), &'static str>) -> Values<'a, (), &'new str> {
        v
    }
}

#[allow(dead_code)]
fn assert_sync() {
    fn map<T: Sync>(v: &BTreeMap<T, T>) -> impl Sync + '_ {
        v
    }

    fn into_iter<T: Sync>(v: BTreeMap<T, T>) -> impl Sync {
        v.into_iter()
    }

    fn into_keys<T: Sync + Ord>(v: BTreeMap<T, T>) -> impl Sync {
        v.into_keys()
    }

    fn into_values<T: Sync + Ord>(v: BTreeMap<T, T>) -> impl Sync {
        v.into_values()
    }

    fn extract_if<T: Sync + Ord>(v: &mut BTreeMap<T, T>) -> impl Sync + '_ {
        v.extract_if(.., |_, _| false)
    }

    fn iter<T: Sync>(v: &BTreeMap<T, T>) -> impl Sync + '_ {
        v.iter()
    }

    fn iter_mut<T: Sync>(v: &mut BTreeMap<T, T>) -> impl Sync + '_ {
        v.iter_mut()
    }

    fn keys<T: Sync>(v: &BTreeMap<T, T>) -> impl Sync + '_ {
        v.keys()
    }

    fn values<T: Sync>(v: &BTreeMap<T, T>) -> impl Sync + '_ {
        v.values()
    }

    fn values_mut<T: Sync>(v: &mut BTreeMap<T, T>) -> impl Sync + '_ {
        v.values_mut()
    }

    fn range<T: Sync + Ord>(v: &BTreeMap<T, T>) -> impl Sync + '_ {
        v.range(..)
    }

    fn range_mut<T: Sync + Ord>(v: &mut BTreeMap<T, T>) -> impl Sync + '_ {
        v.range_mut(..)
    }

    fn entry<T: Sync + Ord + Default>(v: &mut BTreeMap<T, T>) -> impl Sync + '_ {
        v.entry(Default::default())
    }

    fn occupied_entry<T: Sync + Ord + Default>(v: &mut BTreeMap<T, T>) -> impl Sync + '_ {
        match v.entry(Default::default()) {
            Occupied(entry) => entry,
            _ => unreachable!(),
        }
    }

    fn vacant_entry<T: Sync + Ord + Default>(v: &mut BTreeMap<T, T>) -> impl Sync + '_ {
        match v.entry(Default::default()) {
            Vacant(entry) => entry,
            _ => unreachable!(),
        }
    }
}

#[allow(dead_code)]
fn assert_send() {
    fn map<T: Send>(v: BTreeMap<T, T>) -> impl Send {
        v
    }

    fn into_iter<T: Send>(v: BTreeMap<T, T>) -> impl Send {
        v.into_iter()
    }

    fn into_keys<T: Send + Ord>(v: BTreeMap<T, T>) -> impl Send {
        v.into_keys()
    }

    fn into_values<T: Send + Ord>(v: BTreeMap<T, T>) -> impl Send {
        v.into_values()
    }

    fn extract_if<T: Send + Ord>(v: &mut BTreeMap<T, T>) -> impl Send + '_ {
        v.extract_if(.., |_, _| false)
    }

    fn iter<T: Send + Sync>(v: &BTreeMap<T, T>) -> impl Send + '_ {
        v.iter()
    }

    fn iter_mut<T: Send>(v: &mut BTreeMap<T, T>) -> impl Send + '_ {
        v.iter_mut()
    }

    fn keys<T: Send + Sync>(v: &BTreeMap<T, T>) -> impl Send + '_ {
        v.keys()
    }

    fn values<T: Send + Sync>(v: &BTreeMap<T, T>) -> impl Send + '_ {
        v.values()
    }

    fn values_mut<T: Send>(v: &mut BTreeMap<T, T>) -> impl Send + '_ {
        v.values_mut()
    }

    fn range<T: Send + Sync + Ord>(v: &BTreeMap<T, T>) -> impl Send + '_ {
        v.range(..)
    }

    fn range_mut<T: Send + Ord>(v: &mut BTreeMap<T, T>) -> impl Send + '_ {
        v.range_mut(..)
    }

    fn entry<T: Send + Ord + Default>(v: &mut BTreeMap<T, T>) -> impl Send + '_ {
        v.entry(Default::default())
    }

    fn occupied_entry<T: Send + Ord + Default>(v: &mut BTreeMap<T, T>) -> impl Send + '_ {
        match v.entry(Default::default()) {
            Occupied(entry) => entry,
            _ => unreachable!(),
        }
    }

    fn vacant_entry<T: Send + Ord + Default>(v: &mut BTreeMap<T, T>) -> impl Send + '_ {
        match v.entry(Default::default()) {
            Vacant(entry) => entry,
            _ => unreachable!(),
        }
    }
}

#[test]
fn test_ord_absence() {
    fn map<K>(mut map: BTreeMap<K, ()>) {
        let _ = map.is_empty();
        let _ = map.len();
        map.clear();
        let _ = map.iter();
        let _ = map.iter_mut();
        let _ = map.keys();
        let _ = map.values();
        let _ = map.values_mut();
        if true {
            let _ = map.into_values();
        } else if true {
            let _ = map.into_iter();
        } else {
            let _ = map.into_keys();
        }
    }

    fn map_debug<K: Debug>(mut map: BTreeMap<K, ()>) {
        let _ = format!("{map:?}");
        let _ = format!("{:?}", map.iter());
        let _ = format!("{:?}", map.iter_mut());
        let _ = format!("{:?}", map.keys());
        let _ = format!("{:?}", map.values());
        let _ = format!("{:?}", map.values_mut());
        if true {
            let _ = format!("{:?}", map.into_iter());
        } else if true {
            let _ = format!("{:?}", map.into_keys());
        } else {
            let _ = format!("{:?}", map.into_values());
        }
    }

    fn map_clone<K: Clone>(mut map: BTreeMap<K, ()>) {
        map.clone_from(&map.clone());
    }

    #[derive(Debug, Clone)]
    struct NonOrd;
    map(BTreeMap::<NonOrd, _>::new());
    map_debug(BTreeMap::<NonOrd, _>::new());
    map_clone(BTreeMap::<NonOrd, _>::default());
}

#[test]
fn test_occupied_entry_key() {
    let mut a = BTreeMap::new();
    let key = "hello there";
    let value = "value goes here";
    assert_eq!(a.height(), None);
    a.insert(key, value);
    assert_eq!(a.len(), 1);
    assert_eq!(a[key], value);

    match a.entry(key) {
        Vacant(_) => panic!(),
        Occupied(e) => assert_eq!(key, *e.key()),
    }
    assert_eq!(a.len(), 1);
    assert_eq!(a[key], value);
    a.check();
}

#[test]
fn test_vacant_entry_key() {
    let mut a = BTreeMap::new();
    let key = "hello there";
    let value = "value goes here";

    assert_eq!(a.height(), None);
    match a.entry(key) {
        Occupied(_) => unreachable!(),
        Vacant(e) => {
            assert_eq!(key, *e.key());
            e.insert(value);
        }
    }
    assert_eq!(a.len(), 1);
    assert_eq!(a[key], value);
    a.check();
}

#[test]
fn test_vacant_entry_no_insert() {
    let mut a = BTreeMap::<&str, ()>::new();
    let key = "hello there";

    // Non-allocated
    assert_eq!(a.height(), None);
    match a.entry(key) {
        Occupied(_) => unreachable!(),
        Vacant(e) => assert_eq!(key, *e.key()),
    }
    // Ensures the tree has no root.
    assert_eq!(a.height(), None);
    a.check();

    // Allocated but still empty
    a.insert(key, ());
    a.remove(&key);
    assert_eq!(a.height(), Some(0));
    assert!(a.is_empty());
    match a.entry(key) {
        Occupied(_) => unreachable!(),
        Vacant(e) => assert_eq!(key, *e.key()),
    }
    // Ensures the allocated root is not changed.
    assert_eq!(a.height(), Some(0));
    assert!(a.is_empty());
    a.check();
}

#[test]
fn test_first_last_entry() {
    let mut a = BTreeMap::new();
    assert!(a.first_entry().is_none());
    assert!(a.last_entry().is_none());
    a.insert(1, 42);
    assert_eq!(a.first_entry().unwrap().key(), &1);
    assert_eq!(a.last_entry().unwrap().key(), &1);
    a.insert(2, 24);
    assert_eq!(a.first_entry().unwrap().key(), &1);
    assert_eq!(a.last_entry().unwrap().key(), &2);
    a.insert(0, 6);
    assert_eq!(a.first_entry().unwrap().key(), &0);
    assert_eq!(a.last_entry().unwrap().key(), &2);
    let (k1, v1) = a.first_entry().unwrap().remove_entry();
    assert_eq!(k1, 0);
    assert_eq!(v1, 6);
    let (k2, v2) = a.last_entry().unwrap().remove_entry();
    assert_eq!(k2, 2);
    assert_eq!(v2, 24);
    assert_eq!(a.first_entry().unwrap().key(), &1);
    assert_eq!(a.last_entry().unwrap().key(), &1);
    a.check();
}

#[test]
fn test_pop_first_last() {
    let mut map = BTreeMap::new();
    assert_eq!(map.pop_first(), None);
    assert_eq!(map.pop_last(), None);

    map.insert(1, 10);
    map.insert(2, 20);
    map.insert(3, 30);
    map.insert(4, 40);

    assert_eq!(map.len(), 4);

    let (key, val) = map.pop_first().unwrap();
    assert_eq!(key, 1);
    assert_eq!(val, 10);
    assert_eq!(map.len(), 3);

    let (key, val) = map.pop_first().unwrap();
    assert_eq!(key, 2);
    assert_eq!(val, 20);
    assert_eq!(map.len(), 2);
    let (key, val) = map.pop_last().unwrap();
    assert_eq!(key, 4);
    assert_eq!(val, 40);
    assert_eq!(map.len(), 1);

    map.insert(5, 50);
    map.insert(6, 60);
    assert_eq!(map.len(), 3);

    let (key, val) = map.pop_first().unwrap();
    assert_eq!(key, 3);
    assert_eq!(val, 30);
    assert_eq!(map.len(), 2);

    let (key, val) = map.pop_last().unwrap();
    assert_eq!(key, 6);
    assert_eq!(val, 60);
    assert_eq!(map.len(), 1);

    let (key, val) = map.pop_last().unwrap();
    assert_eq!(key, 5);
    assert_eq!(val, 50);
    assert_eq!(map.len(), 0);

    assert_eq!(map.pop_first(), None);
    assert_eq!(map.pop_last(), None);

    map.insert(7, 70);
    map.insert(8, 80);

    let (key, val) = map.pop_last().unwrap();
    assert_eq!(key, 8);
    assert_eq!(val, 80);
    assert_eq!(map.len(), 1);

    let (key, val) = map.pop_last().unwrap();
    assert_eq!(key, 7);
    assert_eq!(val, 70);
    assert_eq!(map.len(), 0);

    assert_eq!(map.pop_first(), None);
    assert_eq!(map.pop_last(), None);
}

#[test]
fn test_get_key_value() {
    let mut map = BTreeMap::new();

    assert!(map.is_empty());
    assert_eq!(map.get_key_value(&1), None);
    assert_eq!(map.get_key_value(&2), None);

    map.insert(1, 10);
    map.insert(2, 20);
    map.insert(3, 30);

    assert_eq!(map.len(), 3);
    assert_eq!(map.get_key_value(&1), Some((&1, &10)));
    assert_eq!(map.get_key_value(&3), Some((&3, &30)));
    assert_eq!(map.get_key_value(&4), None);

    map.remove(&3);

    assert_eq!(map.len(), 2);
    assert_eq!(map.get_key_value(&3), None);
    assert_eq!(map.get_key_value(&2), Some((&2, &20)));
}

#[test]
fn test_insert_into_full_height_0() {
    let size = node::CAPACITY;
    for pos in 0..=size {
        let mut map = BTreeMap::from_iter((0..size).map(|i| (i * 2 + 1, ())));
        assert!(map.insert(pos * 2, ()).is_none());
        map.check();
    }
}

#[test]
fn test_insert_into_full_height_1() {
    let size = node::CAPACITY + 1 + node::CAPACITY;
    for pos in 0..=size {
        let mut map = BTreeMap::from_iter((0..size).map(|i| (i * 2 + 1, ())));
        map.compact();
        let root_node = map.root.as_ref().unwrap().reborrow();
        assert_eq!(root_node.len(), 1);
        assert_eq!(root_node.first_leaf_edge().into_node().len(), node::CAPACITY);
        assert_eq!(root_node.last_leaf_edge().into_node().len(), node::CAPACITY);

        assert!(map.insert(pos * 2, ()).is_none());
        map.check();
    }
}

#[test]
fn test_try_insert() {
    let mut map = BTreeMap::new();

    assert!(map.is_empty());

    assert_eq!(map.try_insert(1, 10).unwrap(), &10);
    assert_eq!(map.try_insert(2, 20).unwrap(), &20);

    let err = map.try_insert(2, 200).unwrap_err();
    assert_eq!(err.entry.key(), &2);
    assert_eq!(err.entry.get(), &20);
    assert_eq!(err.value, 200);
}

macro_rules! create_append_test {
    ($name:ident, $len:expr) => {
        #[test]
        fn $name() {
            let mut a = BTreeMap::new();
            for i in 0..8 {
                a.insert(i, i);
            }

            let mut b = BTreeMap::new();
            for i in 5..$len {
                b.insert(i, 2 * i);
            }

            a.append(&mut b);

            assert_eq!(a.len(), $len);
            assert_eq!(b.len(), 0);

            for i in 0..$len {
                if i < 5 {
                    assert_eq!(a[&i], i);
                } else {
                    assert_eq!(a[&i], 2 * i);
                }
            }

            a.check();
            assert_eq!(a.remove(&($len - 1)), Some(2 * ($len - 1)));
            assert_eq!(a.insert($len - 1, 20), None);
            a.check();
        }
    };
}

// These are mostly for testing the algorithm that "fixes" the right edge after insertion.
// Single node.
create_append_test!(test_append_9, 9);
// Two leafs that don't need fixing.
create_append_test!(test_append_17, 17);
// Two leafs where the second one ends up underfull and needs stealing at the end.
create_append_test!(test_append_14, 14);
// Two leafs where the second one ends up empty because the insertion finished at the root.
create_append_test!(test_append_12, 12);
// Three levels; insertion finished at the root.
create_append_test!(test_append_144, 144);
// Three levels; insertion finished at leaf while there is an empty node on the second level.
create_append_test!(test_append_145, 145);
// Tests for several randomly chosen sizes.
create_append_test!(test_append_170, 170);
create_append_test!(test_append_181, 181);
#[cfg(not(miri))] // Miri is too slow
create_append_test!(test_append_239, 239);
#[cfg(not(miri))] // Miri is too slow
create_append_test!(test_append_1700, 1700);

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_append_drop_leak() {
    let a = CrashTestDummy::new(0);
    let b = CrashTestDummy::new(1);
    let c = CrashTestDummy::new(2);
    let mut left = BTreeMap::new();
    let mut right = BTreeMap::new();
    left.insert(a.spawn(Panic::Never), ());
    left.insert(b.spawn(Panic::InDrop), ()); // first duplicate key, dropped during append
    left.insert(c.spawn(Panic::Never), ());
    right.insert(b.spawn(Panic::Never), ());
    right.insert(c.spawn(Panic::Never), ());

    catch_unwind(move || left.append(&mut right)).unwrap_err();
    assert_eq!(a.dropped(), 1);
    assert_eq!(b.dropped(), 1); // should be 2 were it not for Rust issue #47949
    assert_eq!(c.dropped(), 2);
}

#[test]
fn test_append_ord_chaos() {
    let mut map1 = BTreeMap::new();
    map1.insert(Cyclic3::A, ());
    map1.insert(Cyclic3::B, ());
    let mut map2 = BTreeMap::new();
    map2.insert(Cyclic3::A, ());
    map2.insert(Cyclic3::B, ());
    map2.insert(Cyclic3::C, ()); // lands first, before A
    map2.insert(Cyclic3::B, ()); // lands first, before C
    map1.check();
    map2.check(); // keys are not unique but still strictly ascending
    assert_eq!(map1.len(), 2);
    assert_eq!(map2.len(), 4);
    map1.append(&mut map2);
    assert_eq!(map1.len(), 5);
    assert_eq!(map2.len(), 0);
    map1.check();
    map2.check();
}

fn rand_data(len: usize) -> Vec<(u32, u32)> {
    let mut rng = DeterministicRng::new();
    Vec::from_iter((0..len).map(|_| (rng.next(), rng.next())))
}

#[test]
fn test_split_off_empty_right() {
    let mut data = rand_data(173);

    let mut map = BTreeMap::from_iter(data.clone());
    let right = map.split_off(&(data.iter().max().unwrap().0 + 1));
    map.check();
    right.check();

    data.sort();
    assert!(map.into_iter().eq(data));
    assert!(right.into_iter().eq(None));
}

#[test]
fn test_split_off_empty_left() {
    let mut data = rand_data(314);

    let mut map = BTreeMap::from_iter(data.clone());
    let right = map.split_off(&data.iter().min().unwrap().0);
    map.check();
    right.check();

    data.sort();
    assert!(map.into_iter().eq(None));
    assert!(right.into_iter().eq(data));
}

// In a tree with 3 levels, if all but a part of the first leaf node is split off,
// make sure fix_top eliminates both top levels.
#[test]
fn test_split_off_tiny_left_height_2() {
    let pairs = (0..MIN_INSERTS_HEIGHT_2).map(|i| (i, i));
    let mut left = BTreeMap::from_iter(pairs.clone());
    let right = left.split_off(&1);
    left.check();
    right.check();
    assert_eq!(left.len(), 1);
    assert_eq!(right.len(), MIN_INSERTS_HEIGHT_2 - 1);
    assert_eq!(*left.first_key_value().unwrap().0, 0);
    assert_eq!(*right.first_key_value().unwrap().0, 1);
}

// In a tree with 3 levels, if only part of the last leaf node is split off,
// make sure fix_top eliminates both top levels.
#[test]
fn test_split_off_tiny_right_height_2() {
    let pairs = (0..MIN_INSERTS_HEIGHT_2).map(|i| (i, i));
    let last = MIN_INSERTS_HEIGHT_2 - 1;
    let mut left = BTreeMap::from_iter(pairs.clone());
    assert_eq!(*left.last_key_value().unwrap().0, last);
    let right = left.split_off(&last);
    left.check();
    right.check();
    assert_eq!(left.len(), MIN_INSERTS_HEIGHT_2 - 1);
    assert_eq!(right.len(), 1);
    assert_eq!(*left.last_key_value().unwrap().0, last - 1);
    assert_eq!(*right.last_key_value().unwrap().0, last);
}

#[test]
fn test_split_off_halfway() {
    let mut rng = DeterministicRng::new();
    for &len in &[node::CAPACITY, 25, 50, 75, 100] {
        let mut data = Vec::from_iter((0..len).map(|_| (rng.next(), ())));
        // Insertion in non-ascending order creates some variation in node length.
        let mut map = BTreeMap::from_iter(data.iter().copied());
        data.sort();
        let small_keys = data.iter().take(len / 2).map(|kv| kv.0);
        let large_keys = data.iter().skip(len / 2).map(|kv| kv.0);
        let split_key = large_keys.clone().next().unwrap();
        let right = map.split_off(&split_key);
        map.check();
        right.check();
        assert!(map.keys().copied().eq(small_keys));
        assert!(right.keys().copied().eq(large_keys));
    }
}

#[test]
fn test_split_off_large_random_sorted() {
    // Miri is too slow
    let mut data = if cfg!(miri) { rand_data(529) } else { rand_data(1529) };
    // special case with maximum height.
    data.sort();

    let mut map = BTreeMap::from_iter(data.clone());
    let key = data[data.len() / 2].0;
    let right = map.split_off(&key);
    map.check();
    right.check();

    assert!(map.into_iter().eq(data.clone().into_iter().filter(|x| x.0 < key)));
    assert!(right.into_iter().eq(data.into_iter().filter(|x| x.0 >= key)));
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_into_iter_drop_leak_height_0() {
    let a = CrashTestDummy::new(0);
    let b = CrashTestDummy::new(1);
    let c = CrashTestDummy::new(2);
    let d = CrashTestDummy::new(3);
    let e = CrashTestDummy::new(4);
    let mut map = BTreeMap::new();
    map.insert("a", a.spawn(Panic::Never));
    map.insert("b", b.spawn(Panic::Never));
    map.insert("c", c.spawn(Panic::Never));
    map.insert("d", d.spawn(Panic::InDrop));
    map.insert("e", e.spawn(Panic::Never));

    catch_unwind(move || drop(map.into_iter())).unwrap_err();

    assert_eq!(a.dropped(), 1);
    assert_eq!(b.dropped(), 1);
    assert_eq!(c.dropped(), 1);
    assert_eq!(d.dropped(), 1);
    assert_eq!(e.dropped(), 1);
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_into_iter_drop_leak_kv_panic_in_key() {
    let a_k = CrashTestDummy::new(0);
    let a_v = CrashTestDummy::new(1);
    let b_k = CrashTestDummy::new(2);
    let b_v = CrashTestDummy::new(3);
    let c_k = CrashTestDummy::new(4);
    let c_v = CrashTestDummy::new(5);
    let mut map = BTreeMap::new();
    map.insert(a_k.spawn(Panic::Never), a_v.spawn(Panic::Never));
    map.insert(b_k.spawn(Panic::InDrop), b_v.spawn(Panic::Never));
    map.insert(c_k.spawn(Panic::Never), c_v.spawn(Panic::Never));

    catch_unwind(move || drop(map.into_iter())).unwrap_err();

    assert_eq!(a_k.dropped(), 1);
    assert_eq!(a_v.dropped(), 1);
    assert_eq!(b_k.dropped(), 1);
    assert_eq!(b_v.dropped(), 1);
    assert_eq!(c_k.dropped(), 1);
    assert_eq!(c_v.dropped(), 1);
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_into_iter_drop_leak_kv_panic_in_val() {
    let a_k = CrashTestDummy::new(0);
    let a_v = CrashTestDummy::new(1);
    let b_k = CrashTestDummy::new(2);
    let b_v = CrashTestDummy::new(3);
    let c_k = CrashTestDummy::new(4);
    let c_v = CrashTestDummy::new(5);
    let mut map = BTreeMap::new();
    map.insert(a_k.spawn(Panic::Never), a_v.spawn(Panic::Never));
    map.insert(b_k.spawn(Panic::Never), b_v.spawn(Panic::InDrop));
    map.insert(c_k.spawn(Panic::Never), c_v.spawn(Panic::Never));

    catch_unwind(move || drop(map.into_iter())).unwrap_err();

    assert_eq!(a_k.dropped(), 1);
    assert_eq!(a_v.dropped(), 1);
    assert_eq!(b_k.dropped(), 1);
    assert_eq!(b_v.dropped(), 1);
    assert_eq!(c_k.dropped(), 1);
    assert_eq!(c_v.dropped(), 1);
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_into_iter_drop_leak_height_1() {
    let size = MIN_INSERTS_HEIGHT_1;
    for panic_point in vec![0, 1, size - 2, size - 1] {
        let dummies = Vec::from_iter((0..size).map(|i| CrashTestDummy::new(i)));
        let map = BTreeMap::from_iter((0..size).map(|i| {
            let panic = if i == panic_point { Panic::InDrop } else { Panic::Never };
            (dummies[i].spawn(Panic::Never), dummies[i].spawn(panic))
        }));
        catch_unwind(move || drop(map.into_iter())).unwrap_err();
        for i in 0..size {
            assert_eq!(dummies[i].dropped(), 2);
        }
    }
}

#[test]
fn test_into_keys() {
    let map = BTreeMap::from([(1, 'a'), (2, 'b'), (3, 'c')]);
    let keys = Vec::from_iter(map.into_keys());

    assert_eq!(keys.len(), 3);
    assert!(keys.contains(&1));
    assert!(keys.contains(&2));
    assert!(keys.contains(&3));
}

#[test]
fn test_into_values() {
    let map = BTreeMap::from([(1, 'a'), (2, 'b'), (3, 'c')]);
    let values = Vec::from_iter(map.into_values());

    assert_eq!(values.len(), 3);
    assert!(values.contains(&'a'));
    assert!(values.contains(&'b'));
    assert!(values.contains(&'c'));
}

#[test]
fn test_insert_remove_intertwined() {
    let loops = if cfg!(miri) { 100 } else { 1_000_000 };
    let mut map = BTreeMap::new();
    let mut i = 1;
    let offset = 165; // somewhat arbitrarily chosen to cover some code paths
    for _ in 0..loops {
        i = (i + offset) & 0xFF;
        map.insert(i, i);
        map.remove(&(0xFF - i));
    }
    map.check();
}

#[test]
fn test_insert_remove_intertwined_ord_chaos() {
    let loops = if cfg!(miri) { 100 } else { 1_000_000 };
    let gov = Governor::new();
    let mut map = BTreeMap::new();
    let mut i = 1;
    let offset = 165; // more arbitrarily copied from above
    for _ in 0..loops {
        i = (i + offset) & 0xFF;
        map.insert(Governed(i, &gov), ());
        map.remove(&Governed(0xFF - i, &gov));
        gov.flip();
    }
    map.check_invariants();
}

#[test]
fn from_array() {
    let map = BTreeMap::from([(1, 2), (3, 4)]);
    let unordered_duplicates = BTreeMap::from([(3, 4), (1, 2), (1, 2)]);
    assert_eq!(map, unordered_duplicates);
}

#[test]
fn test_cursor() {
    let map = BTreeMap::from([(1, 'a'), (2, 'b'), (3, 'c')]);

    let mut cur = map.lower_bound(Bound::Unbounded);
    assert_eq!(cur.peek_next(), Some((&1, &'a')));
    assert_eq!(cur.peek_prev(), None);
    assert_eq!(cur.prev(), None);
    assert_eq!(cur.next(), Some((&1, &'a')));

    assert_eq!(cur.next(), Some((&2, &'b')));

    assert_eq!(cur.peek_next(), Some((&3, &'c')));
    assert_eq!(cur.prev(), Some((&2, &'b')));
    assert_eq!(cur.peek_prev(), Some((&1, &'a')));

    let mut cur = map.upper_bound(Bound::Excluded(&1));
    assert_eq!(cur.peek_prev(), None);
    assert_eq!(cur.next(), Some((&1, &'a')));
    assert_eq!(cur.prev(), Some((&1, &'a')));
}

#[test]
fn test_cursor_mut() {
    let mut map = BTreeMap::from([(1, 'a'), (3, 'c'), (5, 'e')]);
    let mut cur = map.lower_bound_mut(Bound::Excluded(&3));
    assert_eq!(cur.peek_next(), Some((&5, &mut 'e')));
    assert_eq!(cur.peek_prev(), Some((&3, &mut 'c')));

    cur.insert_before(4, 'd').unwrap();
    assert_eq!(cur.peek_next(), Some((&5, &mut 'e')));
    assert_eq!(cur.peek_prev(), Some((&4, &mut 'd')));

    assert_eq!(cur.next(), Some((&5, &mut 'e')));
    assert_eq!(cur.peek_next(), None);
    assert_eq!(cur.peek_prev(), Some((&5, &mut 'e')));
    cur.insert_before(6, 'f').unwrap();
    assert_eq!(cur.peek_next(), None);
    assert_eq!(cur.peek_prev(), Some((&6, &mut 'f')));
    assert_eq!(cur.remove_prev(), Some((6, 'f')));
    assert_eq!(cur.remove_prev(), Some((5, 'e')));
    assert_eq!(cur.remove_next(), None);
    assert_eq!(map, BTreeMap::from([(1, 'a'), (3, 'c'), (4, 'd')]));

    let mut cur = map.upper_bound_mut(Bound::Included(&5));
    assert_eq!(cur.peek_next(), None);
    assert_eq!(cur.prev(), Some((&4, &mut 'd')));
    assert_eq!(cur.peek_next(), Some((&4, &mut 'd')));
    assert_eq!(cur.peek_prev(), Some((&3, &mut 'c')));
    assert_eq!(cur.remove_next(), Some((4, 'd')));
    assert_eq!(map, BTreeMap::from([(1, 'a'), (3, 'c')]));
}

#[test]
fn test_cursor_mut_key() {
    let mut map = BTreeMap::from([(1, 'a'), (3, 'c'), (5, 'e')]);
    let mut cur = unsafe { map.lower_bound_mut(Bound::Excluded(&3)).with_mutable_key() };
    assert_eq!(cur.peek_next(), Some((&mut 5, &mut 'e')));
    assert_eq!(cur.peek_prev(), Some((&mut 3, &mut 'c')));

    cur.insert_before(4, 'd').unwrap();
    assert_eq!(cur.peek_next(), Some((&mut 5, &mut 'e')));
    assert_eq!(cur.peek_prev(), Some((&mut 4, &mut 'd')));

    assert_eq!(cur.next(), Some((&mut 5, &mut 'e')));
    assert_eq!(cur.peek_next(), None);
    assert_eq!(cur.peek_prev(), Some((&mut 5, &mut 'e')));
    cur.insert_before(6, 'f').unwrap();
    assert_eq!(cur.peek_next(), None);
    assert_eq!(cur.peek_prev(), Some((&mut 6, &mut 'f')));
    assert_eq!(cur.remove_prev(), Some((6, 'f')));
    assert_eq!(cur.remove_prev(), Some((5, 'e')));
    assert_eq!(cur.remove_next(), None);
    assert_eq!(map, BTreeMap::from([(1, 'a'), (3, 'c'), (4, 'd')]));

    let mut cur = unsafe { map.upper_bound_mut(Bound::Included(&5)).with_mutable_key() };
    assert_eq!(cur.peek_next(), None);
    assert_eq!(cur.prev(), Some((&mut 4, &mut 'd')));
    assert_eq!(cur.peek_next(), Some((&mut 4, &mut 'd')));
    assert_eq!(cur.peek_prev(), Some((&mut 3, &mut 'c')));
    assert_eq!(cur.remove_next(), Some((4, 'd')));
    assert_eq!(map, BTreeMap::from([(1, 'a'), (3, 'c')]));
}

#[test]
fn test_cursor_empty() {
    let mut map = BTreeMap::new();
    let mut cur = map.lower_bound_mut(Bound::Excluded(&3));
    assert_eq!(cur.peek_next(), None);
    assert_eq!(cur.peek_prev(), None);
    cur.insert_after(0, 0).unwrap();
    assert_eq!(cur.peek_next(), Some((&0, &mut 0)));
    assert_eq!(cur.peek_prev(), None);
    assert_eq!(map, BTreeMap::from([(0, 0)]));
}

#[test]
fn test_cursor_mut_insert_before_1() {
    let mut map = BTreeMap::from([(1, 'a'), (2, 'b'), (3, 'c')]);
    let mut cur = map.upper_bound_mut(Bound::Included(&2));
    cur.insert_before(0, 'd').unwrap_err();
}

#[test]
fn test_cursor_mut_insert_before_2() {
    let mut map = BTreeMap::from([(1, 'a'), (2, 'b'), (3, 'c')]);
    let mut cur = map.upper_bound_mut(Bound::Included(&2));
    cur.insert_before(1, 'd').unwrap_err();
}

#[test]
fn test_cursor_mut_insert_before_3() {
    let mut map = BTreeMap::from([(1, 'a'), (2, 'b'), (3, 'c')]);
    let mut cur = map.upper_bound_mut(Bound::Included(&2));
    cur.insert_before(2, 'd').unwrap_err();
}

#[test]
fn test_cursor_mut_insert_before_4() {
    let mut map = BTreeMap::from([(1, 'a'), (2, 'b'), (3, 'c')]);
    let mut cur = map.upper_bound_mut(Bound::Included(&2));
    cur.insert_before(3, 'd').unwrap_err();
}

#[test]
fn test_cursor_mut_insert_after_1() {
    let mut map = BTreeMap::from([(1, 'a'), (2, 'b'), (3, 'c')]);
    let mut cur = map.upper_bound_mut(Bound::Included(&2));
    cur.insert_after(1, 'd').unwrap_err();
}

#[test]
fn test_cursor_mut_insert_after_2() {
    let mut map = BTreeMap::from([(1, 'a'), (2, 'b'), (3, 'c')]);
    let mut cur = map.upper_bound_mut(Bound::Included(&2));
    cur.insert_after(2, 'd').unwrap_err();
}

#[test]
fn test_cursor_mut_insert_after_3() {
    let mut map = BTreeMap::from([(1, 'a'), (2, 'b'), (3, 'c')]);
    let mut cur = map.upper_bound_mut(Bound::Included(&2));
    cur.insert_after(3, 'd').unwrap_err();
}

#[test]
fn test_cursor_mut_insert_after_4() {
    let mut map = BTreeMap::from([(1, 'a'), (2, 'b'), (3, 'c')]);
    let mut cur = map.upper_bound_mut(Bound::Included(&2));
    cur.insert_after(4, 'd').unwrap_err();
}

#[test]
fn cursor_peek_prev_agrees_with_cursor_mut() {
    let mut map = BTreeMap::from([(1, 1), (2, 2), (3, 3)]);

    let cursor = map.lower_bound(Bound::Excluded(&3));
    assert!(cursor.peek_next().is_none());

    let prev = cursor.peek_prev();
    assert_matches!(prev, Some((&3, _)));

    // Shadow names so the two parts of this test match.
    let mut cursor = map.lower_bound_mut(Bound::Excluded(&3));
    assert!(cursor.peek_next().is_none());

    let prev = cursor.peek_prev();
    assert_matches!(prev, Some((&3, _)));
}
