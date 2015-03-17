// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::VecMap;
use std::collections::vec_map::Entry::{Occupied, Vacant};
use std::hash::{SipHasher, hash};

#[test]
fn test_get_mut() {
    let mut m = VecMap::new();
    assert!(m.insert(1, 12).is_none());
    assert!(m.insert(2, 8).is_none());
    assert!(m.insert(5, 14).is_none());
    let new = 100;
    match m.get_mut(&5) {
        None => panic!(), Some(x) => *x = new
    }
    assert_eq!(m.get(&5), Some(&new));
}

#[test]
fn test_len() {
    let mut map = VecMap::new();
    assert_eq!(map.len(), 0);
    assert!(map.is_empty());
    assert!(map.insert(5, 20).is_none());
    assert_eq!(map.len(), 1);
    assert!(!map.is_empty());
    assert!(map.insert(11, 12).is_none());
    assert_eq!(map.len(), 2);
    assert!(!map.is_empty());
    assert!(map.insert(14, 22).is_none());
    assert_eq!(map.len(), 3);
    assert!(!map.is_empty());
}

#[test]
fn test_clear() {
    let mut map = VecMap::new();
    assert!(map.insert(5, 20).is_none());
    assert!(map.insert(11, 12).is_none());
    assert!(map.insert(14, 22).is_none());
    map.clear();
    assert!(map.is_empty());
    assert!(map.get(&5).is_none());
    assert!(map.get(&11).is_none());
    assert!(map.get(&14).is_none());
}

#[test]
fn test_insert() {
    let mut m = VecMap::new();
    assert_eq!(m.insert(1, 2), None);
    assert_eq!(m.insert(1, 3), Some(2));
    assert_eq!(m.insert(1, 4), Some(3));
}

#[test]
fn test_remove() {
    let mut m = VecMap::new();
    m.insert(1, 2);
    assert_eq!(m.remove(&1), Some(2));
    assert_eq!(m.remove(&1), None);
}

#[test]
fn test_keys() {
    let mut map = VecMap::new();
    map.insert(1, 'a');
    map.insert(2, 'b');
    map.insert(3, 'c');
    let keys: Vec<_> = map.keys().collect();
    assert_eq!(keys.len(), 3);
    assert!(keys.contains(&1));
    assert!(keys.contains(&2));
    assert!(keys.contains(&3));
}

#[test]
fn test_values() {
    let mut map = VecMap::new();
    map.insert(1, 'a');
    map.insert(2, 'b');
    map.insert(3, 'c');
    let values: Vec<_> = map.values().cloned().collect();
    assert_eq!(values.len(), 3);
    assert!(values.contains(&'a'));
    assert!(values.contains(&'b'));
    assert!(values.contains(&'c'));
}

#[test]
fn test_iterator() {
    let mut m = VecMap::new();

    assert!(m.insert(0, 1).is_none());
    assert!(m.insert(1, 2).is_none());
    assert!(m.insert(3, 5).is_none());
    assert!(m.insert(6, 10).is_none());
    assert!(m.insert(10, 11).is_none());

    let mut it = m.iter();
    assert_eq!(it.size_hint(), (0, Some(11)));
    assert_eq!(it.next().unwrap(), (0, &1));
    assert_eq!(it.size_hint(), (0, Some(10)));
    assert_eq!(it.next().unwrap(), (1, &2));
    assert_eq!(it.size_hint(), (0, Some(9)));
    assert_eq!(it.next().unwrap(), (3, &5));
    assert_eq!(it.size_hint(), (0, Some(7)));
    assert_eq!(it.next().unwrap(), (6, &10));
    assert_eq!(it.size_hint(), (0, Some(4)));
    assert_eq!(it.next().unwrap(), (10, &11));
    assert_eq!(it.size_hint(), (0, Some(0)));
    assert!(it.next().is_none());
}

#[test]
fn test_iterator_size_hints() {
    let mut m = VecMap::new();

    assert!(m.insert(0, 1).is_none());
    assert!(m.insert(1, 2).is_none());
    assert!(m.insert(3, 5).is_none());
    assert!(m.insert(6, 10).is_none());
    assert!(m.insert(10, 11).is_none());

    assert_eq!(m.iter().size_hint(), (0, Some(11)));
    assert_eq!(m.iter().rev().size_hint(), (0, Some(11)));
    assert_eq!(m.iter_mut().size_hint(), (0, Some(11)));
    assert_eq!(m.iter_mut().rev().size_hint(), (0, Some(11)));
}

#[test]
fn test_mut_iterator() {
    let mut m = VecMap::new();

    assert!(m.insert(0, 1).is_none());
    assert!(m.insert(1, 2).is_none());
    assert!(m.insert(3, 5).is_none());
    assert!(m.insert(6, 10).is_none());
    assert!(m.insert(10, 11).is_none());

    for (k, v) in &mut m {
        *v += k as isize;
    }

    let mut it = m.iter();
    assert_eq!(it.next().unwrap(), (0, &1));
    assert_eq!(it.next().unwrap(), (1, &3));
    assert_eq!(it.next().unwrap(), (3, &8));
    assert_eq!(it.next().unwrap(), (6, &16));
    assert_eq!(it.next().unwrap(), (10, &21));
    assert!(it.next().is_none());
}

#[test]
fn test_rev_iterator() {
    let mut m = VecMap::new();

    assert!(m.insert(0, 1).is_none());
    assert!(m.insert(1, 2).is_none());
    assert!(m.insert(3, 5).is_none());
    assert!(m.insert(6, 10).is_none());
    assert!(m.insert(10, 11).is_none());

    let mut it = m.iter().rev();
    assert_eq!(it.next().unwrap(), (10, &11));
    assert_eq!(it.next().unwrap(), (6, &10));
    assert_eq!(it.next().unwrap(), (3, &5));
    assert_eq!(it.next().unwrap(), (1, &2));
    assert_eq!(it.next().unwrap(), (0, &1));
    assert!(it.next().is_none());
}

#[test]
fn test_mut_rev_iterator() {
    let mut m = VecMap::new();

    assert!(m.insert(0, 1).is_none());
    assert!(m.insert(1, 2).is_none());
    assert!(m.insert(3, 5).is_none());
    assert!(m.insert(6, 10).is_none());
    assert!(m.insert(10, 11).is_none());

    for (k, v) in m.iter_mut().rev() {
        *v += k as isize;
    }

    let mut it = m.iter();
    assert_eq!(it.next().unwrap(), (0, &1));
    assert_eq!(it.next().unwrap(), (1, &3));
    assert_eq!(it.next().unwrap(), (3, &8));
    assert_eq!(it.next().unwrap(), (6, &16));
    assert_eq!(it.next().unwrap(), (10, &21));
    assert!(it.next().is_none());
}

#[test]
fn test_move_iter() {
    let mut m: VecMap<Box<_>> = VecMap::new();
    m.insert(1, box 2);
    let mut called = false;
    for (k, v) in m {
        assert!(!called);
        called = true;
        assert_eq!(k, 1);
        assert_eq!(v, box 2);
    }
    assert!(called);
}

#[test]
fn test_drain_iterator() {
    let mut map = VecMap::new();
    map.insert(1, "a");
    map.insert(3, "c");
    map.insert(2, "b");

    let vec: Vec<_> = map.drain().collect();

    assert_eq!(vec, [(1, "a"), (2, "b"), (3, "c")]);
    assert_eq!(map.len(), 0);
}

#[test]
fn test_append() {
    let mut a = VecMap::new();
    a.insert(1, "a");
    a.insert(2, "b");
    a.insert(3, "c");

    let mut b = VecMap::new();
    b.insert(3, "d");  // Overwrite element from a
    b.insert(4, "e");
    b.insert(5, "f");

    a.append(&mut b);

    assert_eq!(a.len(), 5);
    assert_eq!(b.len(), 0);
    // Capacity shouldn't change for possible reuse
    assert!(b.capacity() >= 4);

    assert_eq!(a[1], "a");
    assert_eq!(a[2], "b");
    assert_eq!(a[3], "d");
    assert_eq!(a[4], "e");
    assert_eq!(a[5], "f");
}

#[test]
fn test_split_off() {
    // Split within the key range
    let mut a = VecMap::new();
    a.insert(1, "a");
    a.insert(2, "b");
    a.insert(3, "c");
    a.insert(4, "d");

    let b = a.split_off(3);

    assert_eq!(a.len(), 2);
    assert_eq!(b.len(), 2);

    assert_eq!(a[1], "a");
    assert_eq!(a[2], "b");

    assert_eq!(b[3], "c");
    assert_eq!(b[4], "d");

    // Split at 0
    a.clear();
    a.insert(1, "a");
    a.insert(2, "b");
    a.insert(3, "c");
    a.insert(4, "d");

    let b = a.split_off(0);

    assert_eq!(a.len(), 0);
    assert_eq!(b.len(), 4);
    assert_eq!(b[1], "a");
    assert_eq!(b[2], "b");
    assert_eq!(b[3], "c");
    assert_eq!(b[4], "d");

    // Split behind max_key
    a.clear();
    a.insert(1, "a");
    a.insert(2, "b");
    a.insert(3, "c");
    a.insert(4, "d");

    let b = a.split_off(5);

    assert_eq!(a.len(), 4);
    assert_eq!(b.len(), 0);
    assert_eq!(a[1], "a");
    assert_eq!(a[2], "b");
    assert_eq!(a[3], "c");
    assert_eq!(a[4], "d");
}

#[test]
fn test_show() {
    let mut map = VecMap::new();
    let empty = VecMap::<i32>::new();

    map.insert(1, 2);
    map.insert(3, 4);

    let map_str = format!("{:?}", map);
    assert!(map_str == "{1: 2, 3: 4}" || map_str == "{3: 4, 1: 2}");
    assert_eq!(format!("{:?}", empty), "{}");
}

#[test]
fn test_clone() {
    let mut a = VecMap::new();

    a.insert(1, 'x');
    a.insert(4, 'y');
    a.insert(6, 'z');

    assert!(a.clone() == a);
}

#[test]
fn test_eq() {
    let mut a = VecMap::new();
    let mut b = VecMap::new();

    assert!(a == b);
    assert!(a.insert(0, 5).is_none());
    assert!(a != b);
    assert!(b.insert(0, 4).is_none());
    assert!(a != b);
    assert!(a.insert(5, 19).is_none());
    assert!(a != b);
    assert!(!b.insert(0, 5).is_none());
    assert!(a != b);
    assert!(b.insert(5, 19).is_none());
    assert!(a == b);

    a = VecMap::new();
    b = VecMap::with_capacity(1);
    assert!(a == b);
}

#[test]
fn test_lt() {
    let mut a = VecMap::new();
    let mut b = VecMap::new();

    assert!(!(a < b) && !(b < a));
    assert!(b.insert(2, 5).is_none());
    assert!(a < b);
    assert!(a.insert(2, 7).is_none());
    assert!(!(a < b) && b < a);
    assert!(b.insert(1, 0).is_none());
    assert!(b < a);
    assert!(a.insert(0, 6).is_none());
    assert!(a < b);
    assert!(a.insert(6, 2).is_none());
    assert!(a < b && !(b < a));
}

#[test]
fn test_ord() {
    let mut a = VecMap::new();
    let mut b = VecMap::new();

    assert!(a <= b && a >= b);
    assert!(a.insert(1, 1).is_none());
    assert!(a > b && a >= b);
    assert!(b < a && b <= a);
    assert!(b.insert(2, 2).is_none());
    assert!(b > a && b >= a);
    assert!(a < b && a <= b);
}

#[test]
fn test_hash() {
    let mut x = VecMap::new();
    let mut y = VecMap::new();

    assert!(hash::<_, SipHasher>(&x) == hash::<_, SipHasher>(&y));
    x.insert(1, 'a');
    x.insert(2, 'b');
    x.insert(3, 'c');

    y.insert(3, 'c');
    y.insert(2, 'b');
    y.insert(1, 'a');

    assert!(hash::<_, SipHasher>(&x) == hash::<_, SipHasher>(&y));

    x.insert(1000, 'd');
    x.remove(&1000);

    assert!(hash::<_, SipHasher>(&x) == hash::<_, SipHasher>(&y));
}

#[test]
fn test_from_iter() {
    let xs = vec![(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e')];

    let map: VecMap<_> = xs.iter().cloned().collect();

    for &(k, v) in &xs {
        assert_eq!(map.get(&k), Some(&v));
    }
}

#[test]
fn test_index() {
    let mut map = VecMap::new();

    map.insert(1, 2);
    map.insert(2, 1);
    map.insert(3, 4);

    assert_eq!(map[3], 4);
}

#[test]
#[should_panic]
fn test_index_nonexistent() {
    let mut map = VecMap::new();

    map.insert(1, 2);
    map.insert(2, 1);
    map.insert(3, 4);

    map[4];
}

#[test]
fn test_entry(){
    let xs = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

    let mut map: VecMap<_> = xs.iter().cloned().collect();

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

mod bench {
    use std::collections::VecMap;

    map_insert_rand_bench!{insert_rand_100,    100,    VecMap}
    map_insert_rand_bench!{insert_rand_10_000, 10_000, VecMap}

    map_insert_seq_bench!{insert_seq_100,    100,    VecMap}
    map_insert_seq_bench!{insert_seq_10_000, 10_000, VecMap}

    map_find_rand_bench!{find_rand_100,    100,    VecMap}
    map_find_rand_bench!{find_rand_10_000, 10_000, VecMap}

    map_find_seq_bench!{find_seq_100,    100,    VecMap}
    map_find_seq_bench!{find_seq_10_000, 10_000, VecMap}
}
