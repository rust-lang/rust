use std::collections::BTreeMap;
use std::collections::btree_map::Entry::{Occupied, Vacant};
use std::ops::Bound::{self, Excluded, Included, Unbounded};
use std::rc::Rc;
use std::iter::FromIterator;

use super::DeterministicRng;

#[test]
fn test_basic_large() {
    let mut map = BTreeMap::new();
    #[cfg(not(miri))] // Miri is too slow
    let size = 10000;
    #[cfg(miri)]
    let size = 200;
    assert_eq!(map.len(), 0);

    for i in 0..size {
        assert_eq!(map.insert(i, 10 * i), None);
        assert_eq!(map.len(), i + 1);
    }

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
}

#[test]
fn test_basic_small() {
    let mut map = BTreeMap::new();
    assert_eq!(map.remove(&1), None);
    assert_eq!(map.get(&1), None);
    assert_eq!(map.insert(1, 1), None);
    assert_eq!(map.get(&1), Some(&1));
    assert_eq!(map.insert(1, 2), Some(1));
    assert_eq!(map.get(&1), Some(&2));
    assert_eq!(map.insert(2, 4), None);
    assert_eq!(map.get(&2), Some(&4));
    assert_eq!(map.remove(&1), Some(2));
    assert_eq!(map.remove(&2), Some(4));
    assert_eq!(map.remove(&1), None);
}

#[test]
fn test_iter() {
    #[cfg(not(miri))] // Miri is too slow
    let size = 10000;
    #[cfg(miri)]
    let size = 200;

    // Forwards
    let mut map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();

    fn test<T>(size: usize, mut iter: T)
        where T: Iterator<Item = (usize, usize)>
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
    #[cfg(not(miri))] // Miri is too slow
    let size = 10000;
    #[cfg(miri)]
    let size = 200;

    // Forwards
    let mut map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();

    fn test<T>(size: usize, mut iter: T)
        where T: Iterator<Item = (usize, usize)>
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

#[test]
fn test_values_mut() {
    let mut a = BTreeMap::new();
    a.insert(1, String::from("hello"));
    a.insert(2, String::from("goodbye"));

    for value in a.values_mut() {
        value.push_str("!");
    }

    let values: Vec<String> = a.values().cloned().collect();
    assert_eq!(values, [String::from("hello!"), String::from("goodbye!")]);
}

#[test]
fn test_iter_mixed() {
    #[cfg(not(miri))] // Miri is too slow
    let size = 10000;
    #[cfg(miri)]
    let size = 200;

    // Forwards
    let mut map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();

    fn test<T>(size: usize, mut iter: T)
        where T: Iterator<Item = (usize, usize)> + DoubleEndedIterator
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
fn test_range_small() {
    let size = 5;

    // Forwards
    let map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();

    let mut j = 0;
    for ((&k, &v), i) in map.range(2..).zip(2..size) {
        assert_eq!(k, i);
        assert_eq!(v, i);
        j += 1;
    }
    assert_eq!(j, size - 2);
}

#[test]
fn test_range_inclusive() {
    let size = 500;

    let map: BTreeMap<_, _> = (0..=size).map(|i| (i, i)).collect();

    fn check<'a, L, R>(lhs: L, rhs: R)
        where L: IntoIterator<Item=(&'a i32, &'a i32)>,
              R: IntoIterator<Item=(&'a i32, &'a i32)>,
    {
        let lhs: Vec<_> = lhs.into_iter().collect();
        let rhs: Vec<_> = rhs.into_iter().collect();
        assert_eq!(lhs, rhs);
    }

    check(map.range(size + 1..=size + 1), vec![]);
    check(map.range(size..=size), vec![(&size, &size)]);
    check(map.range(size..=size + 1), vec![(&size, &size)]);
    check(map.range(0..=0), vec![(&0, &0)]);
    check(map.range(0..=size - 1), map.range(..size));
    check(map.range(-1..=-1), vec![]);
    check(map.range(-1..=size), map.range(..));
    check(map.range(..=size), map.range(..));
    check(map.range(..=200), map.range(..201));
    check(map.range(5..=8), vec![(&5, &5), (&6, &6), (&7, &7), (&8, &8)]);
    check(map.range(-1..=0), vec![(&0, &0)]);
    check(map.range(-1..=2), vec![(&0, &0), (&1, &1), (&2, &2)]);
}

#[test]
fn test_range_inclusive_max_value() {
    let max = std::usize::MAX;
    let map: BTreeMap<_, _> = vec![(max, 0)].into_iter().collect();

    assert_eq!(map.range(max..=max).collect::<Vec<_>>(), &[(&max, &0)]);
}

#[test]
fn test_range_equal_empty_cases() {
    let map: BTreeMap<_, _> = (0..5).map(|i| (i, i)).collect();
    assert_eq!(map.range((Included(2), Excluded(2))).next(), None);
    assert_eq!(map.range((Excluded(2), Included(2))).next(), None);
}

#[test]
#[should_panic]
fn test_range_equal_excluded() {
    let map: BTreeMap<_, _> = (0..5).map(|i| (i, i)).collect();
    map.range((Excluded(2), Excluded(2)));
}

#[test]
#[should_panic]
fn test_range_backwards_1() {
    let map: BTreeMap<_, _> = (0..5).map(|i| (i, i)).collect();
    map.range((Included(3), Included(2)));
}

#[test]
#[should_panic]
fn test_range_backwards_2() {
    let map: BTreeMap<_, _> = (0..5).map(|i| (i, i)).collect();
    map.range((Included(3), Excluded(2)));
}

#[test]
#[should_panic]
fn test_range_backwards_3() {
    let map: BTreeMap<_, _> = (0..5).map(|i| (i, i)).collect();
    map.range((Excluded(3), Included(2)));
}

#[test]
#[should_panic]
fn test_range_backwards_4() {
    let map: BTreeMap<_, _> = (0..5).map(|i| (i, i)).collect();
    map.range((Excluded(3), Excluded(2)));
}

#[test]
fn test_range_1000() {
    #[cfg(not(miri))] // Miri is too slow
    let size = 1000;
    #[cfg(miri)]
    let size = 200;
    let map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();

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
    let mut iter = map.range::<str, _>((Included("b"),Excluded("d")));
    assert_eq!(iter.next(), Some((&"baboon".to_string(), &2)));
    assert_eq!(iter.next(), Some((&"coyote".to_string(), &3)));
    assert_eq!(iter.next(), None);
}

#[test]
fn test_range() {
    #[cfg(not(miri))] // Miri is too slow
    let size = 200;
    #[cfg(miri)]
    let size = 30;
    let map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();

    for i in 0..size {
        for j in i..size {
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
    #[cfg(not(miri))] // Miri is too slow
    let size = 200;
    #[cfg(miri)]
    let size = 30;
    let mut map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();

    for i in 0..size {
        for j in i..size {
            let mut kvs = map.range_mut((Included(&i), Included(&j))).map(|(&k, &mut v)| (k, v));
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
}

#[test]
fn test_entry() {
    let xs = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

    let mut map: BTreeMap<_, _> = xs.iter().cloned().collect();

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

    // Existing key (take)
    match map.entry(3) {
        Vacant(_) => unreachable!(),
        Occupied(view) => {
            assert_eq!(view.remove(), 30);
        }
    }
    assert_eq!(map.get(&3), None);
    assert_eq!(map.len(), 5);


    // Inexistent key (insert)
    match map.entry(10) {
        Occupied(_) => unreachable!(),
        Vacant(view) => {
            assert_eq!(*view.insert(1000), 1000);
        }
    }
    assert_eq!(map.get(&10).unwrap(), &1000);
    assert_eq!(map.len(), 6);
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
}

// This test's only purpose is to ensure that zero-sized keys with nonsensical orderings
// do not cause segfaults when used with zero-sized values. All other map behavior is
// undefined.
#[test]
fn test_bad_zst() {
    use std::cmp::Ordering;

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
}

#[test]
fn test_clone() {
    let mut map = BTreeMap::new();
    #[cfg(not(miri))] // Miri is too slow
    let size = 100;
    #[cfg(miri)]
    let size = 30;
    assert_eq!(map.len(), 0);

    for i in 0..size {
        assert_eq!(map.insert(i, 10 * i), None);
        assert_eq!(map.len(), i + 1);
        assert_eq!(map, map.clone());
    }

    for i in 0..size {
        assert_eq!(map.insert(i, 100 * i), Some(10 * i));
        assert_eq!(map.len(), size);
        assert_eq!(map, map.clone());
    }

    for i in 0..size / 2 {
        assert_eq!(map.remove(&(i * 2)), Some(i * 200));
        assert_eq!(map.len(), size - i - 1);
        assert_eq!(map, map.clone());
    }

    for i in 0..size / 2 {
        assert_eq!(map.remove(&(2 * i)), None);
        assert_eq!(map.remove(&(2 * i + 1)), Some(i * 200 + 100));
        assert_eq!(map.len(), size / 2 - i - 1);
        assert_eq!(map, map.clone());
    }
}

#[test]
#[allow(dead_code)]
fn test_variance() {
    use std::collections::btree_map::{Iter, IntoIter, Range, Keys, Values};

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
    fn range_key<'a, 'new>(v: Range<'a, &'static str, ()>) -> Range<'a, &'new str, ()> {
        v
    }
    fn range_val<'a, 'new>(v: Range<'a, (), &'static str>) -> Range<'a, (), &'new str> {
        v
    }
    fn keys<'a, 'new>(v: Keys<'a, &'static str, ()>) -> Keys<'a, &'new str, ()> {
        v
    }
    fn vals<'a, 'new>(v: Values<'a, (), &'static str>) -> Values<'a, (), &'new str> {
        v
    }
}

#[test]
fn test_occupied_entry_key() {
    let mut a = BTreeMap::new();
    let key = "hello there";
    let value = "value goes here";
    assert!(a.is_empty());
    a.insert(key.clone(), value.clone());
    assert_eq!(a.len(), 1);
    assert_eq!(a[key], value);

    match a.entry(key.clone()) {
        Vacant(_) => panic!(),
        Occupied(e) => assert_eq!(key, *e.key()),
    }
    assert_eq!(a.len(), 1);
    assert_eq!(a[key], value);
}

#[test]
fn test_vacant_entry_key() {
    let mut a = BTreeMap::new();
    let key = "hello there";
    let value = "value goes here";

    assert!(a.is_empty());
    match a.entry(key.clone()) {
        Occupied(_) => panic!(),
        Vacant(e) => {
            assert_eq!(key, *e.key());
            e.insert(value.clone());
        }
    }
    assert_eq!(a.len(), 1);
    assert_eq!(a[key], value);
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
                b.insert(i, 2*i);
            }

            a.append(&mut b);

            assert_eq!(a.len(), $len);
            assert_eq!(b.len(), 0);

            for i in 0..$len {
                if i < 5 {
                    assert_eq!(a[&i], i);
                } else {
                    assert_eq!(a[&i], 2*i);
                }
            }

            assert_eq!(a.remove(&($len-1)), Some(2*($len-1)));
            assert_eq!(a.insert($len-1, 20), None);
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
create_append_test!(test_append_239, 239);
#[cfg(not(miri))] // Miri is too slow
create_append_test!(test_append_1700, 1700);

fn rand_data(len: usize) -> Vec<(u32, u32)> {
    let mut rng = DeterministicRng::new();
    Vec::from_iter((0..len).map(|_| (rng.next(), rng.next())))
}

#[test]
fn test_split_off_empty_right() {
    let mut data = rand_data(173);

    let mut map = BTreeMap::from_iter(data.clone());
    let right = map.split_off(&(data.iter().max().unwrap().0 + 1));

    data.sort();
    assert!(map.into_iter().eq(data));
    assert!(right.into_iter().eq(None));
}

#[test]
fn test_split_off_empty_left() {
    let mut data = rand_data(314);

    let mut map = BTreeMap::from_iter(data.clone());
    let right = map.split_off(&data.iter().min().unwrap().0);

    data.sort();
    assert!(map.into_iter().eq(None));
    assert!(right.into_iter().eq(data));
}

#[test]
fn test_split_off_large_random_sorted() {
    let mut data = rand_data(1529);
    // special case with maximum height.
    data.sort();

    let mut map = BTreeMap::from_iter(data.clone());
    let key = data[data.len() / 2].0;
    let right = map.split_off(&key);

    assert!(map.into_iter().eq(data.clone().into_iter().filter(|x| x.0 < key)));
    assert!(right.into_iter().eq(data.into_iter().filter(|x| x.0 >= key)));
}
