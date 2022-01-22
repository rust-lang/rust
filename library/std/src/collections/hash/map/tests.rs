use super::Entry::{Occupied, Vacant};
use super::HashMap;
use super::RandomState;
use crate::assert_matches::assert_matches;
use crate::cell::RefCell;
use rand::{thread_rng, Rng};
use realstd::collections::TryReserveErrorKind::*;

// https://github.com/rust-lang/rust/issues/62301
fn _assert_hashmap_is_unwind_safe() {
    fn assert_unwind_safe<T: crate::panic::UnwindSafe>() {}
    assert_unwind_safe::<HashMap<(), crate::cell::UnsafeCell<()>>>();
}

#[test]
fn test_zero_capacities() {
    type HM = HashMap<i32, i32>;

    let m = HM::new();
    assert_eq!(m.capacity(), 0);

    let m = HM::default();
    assert_eq!(m.capacity(), 0);

    let m = HM::with_hasher(RandomState::new());
    assert_eq!(m.capacity(), 0);

    let m = HM::with_capacity(0);
    assert_eq!(m.capacity(), 0);

    let m = HM::with_capacity_and_hasher(0, RandomState::new());
    assert_eq!(m.capacity(), 0);

    let mut m = HM::new();
    m.insert(1, 1);
    m.insert(2, 2);
    m.remove(&1);
    m.remove(&2);
    m.shrink_to_fit();
    assert_eq!(m.capacity(), 0);

    let mut m = HM::new();
    m.reserve(0);
    assert_eq!(m.capacity(), 0);
}

#[test]
fn test_create_capacity_zero() {
    let mut m = HashMap::with_capacity(0);

    assert!(m.insert(1, 1).is_none());

    assert!(m.contains_key(&1));
    assert!(!m.contains_key(&0));
}

#[test]
fn test_insert() {
    let mut m = HashMap::new();
    assert_eq!(m.len(), 0);
    assert!(m.insert(1, 2).is_none());
    assert_eq!(m.len(), 1);
    assert!(m.insert(2, 4).is_none());
    assert_eq!(m.len(), 2);
    assert_eq!(*m.get(&1).unwrap(), 2);
    assert_eq!(*m.get(&2).unwrap(), 4);
}

#[test]
fn test_clone() {
    let mut m = HashMap::new();
    assert_eq!(m.len(), 0);
    assert!(m.insert(1, 2).is_none());
    assert_eq!(m.len(), 1);
    assert!(m.insert(2, 4).is_none());
    assert_eq!(m.len(), 2);
    let m2 = m.clone();
    assert_eq!(*m2.get(&1).unwrap(), 2);
    assert_eq!(*m2.get(&2).unwrap(), 4);
    assert_eq!(m2.len(), 2);
}

thread_local! { static DROP_VECTOR: RefCell<Vec<i32>> = RefCell::new(Vec::new()) }

#[derive(Hash, PartialEq, Eq)]
struct Droppable {
    k: usize,
}

impl Droppable {
    fn new(k: usize) -> Droppable {
        DROP_VECTOR.with(|slot| {
            slot.borrow_mut()[k] += 1;
        });

        Droppable { k }
    }
}

impl Drop for Droppable {
    fn drop(&mut self) {
        DROP_VECTOR.with(|slot| {
            slot.borrow_mut()[self.k] -= 1;
        });
    }
}

impl Clone for Droppable {
    fn clone(&self) -> Droppable {
        Droppable::new(self.k)
    }
}

#[test]
fn test_drops() {
    DROP_VECTOR.with(|slot| {
        *slot.borrow_mut() = vec![0; 200];
    });

    {
        let mut m = HashMap::new();

        DROP_VECTOR.with(|v| {
            for i in 0..200 {
                assert_eq!(v.borrow()[i], 0);
            }
        });

        for i in 0..100 {
            let d1 = Droppable::new(i);
            let d2 = Droppable::new(i + 100);
            m.insert(d1, d2);
        }

        DROP_VECTOR.with(|v| {
            for i in 0..200 {
                assert_eq!(v.borrow()[i], 1);
            }
        });

        for i in 0..50 {
            let k = Droppable::new(i);
            let v = m.remove(&k);

            assert!(v.is_some());

            DROP_VECTOR.with(|v| {
                assert_eq!(v.borrow()[i], 1);
                assert_eq!(v.borrow()[i + 100], 1);
            });
        }

        DROP_VECTOR.with(|v| {
            for i in 0..50 {
                assert_eq!(v.borrow()[i], 0);
                assert_eq!(v.borrow()[i + 100], 0);
            }

            for i in 50..100 {
                assert_eq!(v.borrow()[i], 1);
                assert_eq!(v.borrow()[i + 100], 1);
            }
        });
    }

    DROP_VECTOR.with(|v| {
        for i in 0..200 {
            assert_eq!(v.borrow()[i], 0);
        }
    });
}

#[test]
fn test_into_iter_drops() {
    DROP_VECTOR.with(|v| {
        *v.borrow_mut() = vec![0; 200];
    });

    let hm = {
        let mut hm = HashMap::new();

        DROP_VECTOR.with(|v| {
            for i in 0..200 {
                assert_eq!(v.borrow()[i], 0);
            }
        });

        for i in 0..100 {
            let d1 = Droppable::new(i);
            let d2 = Droppable::new(i + 100);
            hm.insert(d1, d2);
        }

        DROP_VECTOR.with(|v| {
            for i in 0..200 {
                assert_eq!(v.borrow()[i], 1);
            }
        });

        hm
    };

    // By the way, ensure that cloning doesn't screw up the dropping.
    drop(hm.clone());

    {
        let mut half = hm.into_iter().take(50);

        DROP_VECTOR.with(|v| {
            for i in 0..200 {
                assert_eq!(v.borrow()[i], 1);
            }
        });

        for _ in half.by_ref() {}

        DROP_VECTOR.with(|v| {
            let nk = (0..100).filter(|&i| v.borrow()[i] == 1).count();

            let nv = (0..100).filter(|&i| v.borrow()[i + 100] == 1).count();

            assert_eq!(nk, 50);
            assert_eq!(nv, 50);
        });
    };

    DROP_VECTOR.with(|v| {
        for i in 0..200 {
            assert_eq!(v.borrow()[i], 0);
        }
    });
}

#[test]
fn test_empty_remove() {
    let mut m: HashMap<i32, bool> = HashMap::new();
    assert_eq!(m.remove(&0), None);
}

#[test]
fn test_empty_entry() {
    let mut m: HashMap<i32, bool> = HashMap::new();
    match m.entry(0) {
        Occupied(_) => panic!(),
        Vacant(_) => {}
    }
    assert!(*m.entry(0).or_insert(true));
    assert_eq!(m.len(), 1);
}

#[test]
fn test_empty_iter() {
    let mut m: HashMap<i32, bool> = HashMap::new();
    assert_eq!(m.drain().next(), None);
    assert_eq!(m.keys().next(), None);
    assert_eq!(m.values().next(), None);
    assert_eq!(m.values_mut().next(), None);
    assert_eq!(m.iter().next(), None);
    assert_eq!(m.iter_mut().next(), None);
    assert_eq!(m.len(), 0);
    assert!(m.is_empty());
    assert_eq!(m.into_iter().next(), None);
}

#[test]
fn test_lots_of_insertions() {
    let mut m = HashMap::new();

    // Try this a few times to make sure we never screw up the hashmap's
    // internal state.
    for _ in 0..10 {
        assert!(m.is_empty());

        for i in 1..1001 {
            assert!(m.insert(i, i).is_none());

            for j in 1..=i {
                let r = m.get(&j);
                assert_eq!(r, Some(&j));
            }

            for j in i + 1..1001 {
                let r = m.get(&j);
                assert_eq!(r, None);
            }
        }

        for i in 1001..2001 {
            assert!(!m.contains_key(&i));
        }

        // remove forwards
        for i in 1..1001 {
            assert!(m.remove(&i).is_some());

            for j in 1..=i {
                assert!(!m.contains_key(&j));
            }

            for j in i + 1..1001 {
                assert!(m.contains_key(&j));
            }
        }

        for i in 1..1001 {
            assert!(!m.contains_key(&i));
        }

        for i in 1..1001 {
            assert!(m.insert(i, i).is_none());
        }

        // remove backwards
        for i in (1..1001).rev() {
            assert!(m.remove(&i).is_some());

            for j in i..1001 {
                assert!(!m.contains_key(&j));
            }

            for j in 1..i {
                assert!(m.contains_key(&j));
            }
        }
    }
}

#[test]
fn test_find_mut() {
    let mut m = HashMap::new();
    assert!(m.insert(1, 12).is_none());
    assert!(m.insert(2, 8).is_none());
    assert!(m.insert(5, 14).is_none());
    let new = 100;
    match m.get_mut(&5) {
        None => panic!(),
        Some(x) => *x = new,
    }
    assert_eq!(m.get(&5), Some(&new));
}

#[test]
fn test_insert_overwrite() {
    let mut m = HashMap::new();
    assert!(m.insert(1, 2).is_none());
    assert_eq!(*m.get(&1).unwrap(), 2);
    assert!(!m.insert(1, 3).is_none());
    assert_eq!(*m.get(&1).unwrap(), 3);
}

#[test]
fn test_insert_conflicts() {
    let mut m = HashMap::with_capacity(4);
    assert!(m.insert(1, 2).is_none());
    assert!(m.insert(5, 3).is_none());
    assert!(m.insert(9, 4).is_none());
    assert_eq!(*m.get(&9).unwrap(), 4);
    assert_eq!(*m.get(&5).unwrap(), 3);
    assert_eq!(*m.get(&1).unwrap(), 2);
}

#[test]
fn test_conflict_remove() {
    let mut m = HashMap::with_capacity(4);
    assert!(m.insert(1, 2).is_none());
    assert_eq!(*m.get(&1).unwrap(), 2);
    assert!(m.insert(5, 3).is_none());
    assert_eq!(*m.get(&1).unwrap(), 2);
    assert_eq!(*m.get(&5).unwrap(), 3);
    assert!(m.insert(9, 4).is_none());
    assert_eq!(*m.get(&1).unwrap(), 2);
    assert_eq!(*m.get(&5).unwrap(), 3);
    assert_eq!(*m.get(&9).unwrap(), 4);
    assert!(m.remove(&1).is_some());
    assert_eq!(*m.get(&9).unwrap(), 4);
    assert_eq!(*m.get(&5).unwrap(), 3);
}

#[test]
fn test_is_empty() {
    let mut m = HashMap::with_capacity(4);
    assert!(m.insert(1, 2).is_none());
    assert!(!m.is_empty());
    assert!(m.remove(&1).is_some());
    assert!(m.is_empty());
}

#[test]
fn test_remove() {
    let mut m = HashMap::new();
    m.insert(1, 2);
    assert_eq!(m.remove(&1), Some(2));
    assert_eq!(m.remove(&1), None);
}

#[test]
fn test_remove_entry() {
    let mut m = HashMap::new();
    m.insert(1, 2);
    assert_eq!(m.remove_entry(&1), Some((1, 2)));
    assert_eq!(m.remove(&1), None);
}

#[test]
fn test_iterate() {
    let mut m = HashMap::with_capacity(4);
    for i in 0..32 {
        assert!(m.insert(i, i * 2).is_none());
    }
    assert_eq!(m.len(), 32);

    let mut observed: u32 = 0;

    for (k, v) in &m {
        assert_eq!(*v, *k * 2);
        observed |= 1 << *k;
    }
    assert_eq!(observed, 0xFFFF_FFFF);
}

#[test]
fn test_keys() {
    let pairs = [(1, 'a'), (2, 'b'), (3, 'c')];
    let map: HashMap<_, _> = pairs.into_iter().collect();
    let keys: Vec<_> = map.keys().cloned().collect();
    assert_eq!(keys.len(), 3);
    assert!(keys.contains(&1));
    assert!(keys.contains(&2));
    assert!(keys.contains(&3));
}

#[test]
fn test_values() {
    let pairs = [(1, 'a'), (2, 'b'), (3, 'c')];
    let map: HashMap<_, _> = pairs.into_iter().collect();
    let values: Vec<_> = map.values().cloned().collect();
    assert_eq!(values.len(), 3);
    assert!(values.contains(&'a'));
    assert!(values.contains(&'b'));
    assert!(values.contains(&'c'));
}

#[test]
fn test_values_mut() {
    let pairs = [(1, 1), (2, 2), (3, 3)];
    let mut map: HashMap<_, _> = pairs.into_iter().collect();
    for value in map.values_mut() {
        *value = (*value) * 2
    }
    let values: Vec<_> = map.values().cloned().collect();
    assert_eq!(values.len(), 3);
    assert!(values.contains(&2));
    assert!(values.contains(&4));
    assert!(values.contains(&6));
}

#[test]
fn test_into_keys() {
    let pairs = [(1, 'a'), (2, 'b'), (3, 'c')];
    let map: HashMap<_, _> = pairs.into_iter().collect();
    let keys: Vec<_> = map.into_keys().collect();

    assert_eq!(keys.len(), 3);
    assert!(keys.contains(&1));
    assert!(keys.contains(&2));
    assert!(keys.contains(&3));
}

#[test]
fn test_into_values() {
    let pairs = [(1, 'a'), (2, 'b'), (3, 'c')];
    let map: HashMap<_, _> = pairs.into_iter().collect();
    let values: Vec<_> = map.into_values().collect();

    assert_eq!(values.len(), 3);
    assert!(values.contains(&'a'));
    assert!(values.contains(&'b'));
    assert!(values.contains(&'c'));
}

#[test]
fn test_find() {
    let mut m = HashMap::new();
    assert!(m.get(&1).is_none());
    m.insert(1, 2);
    match m.get(&1) {
        None => panic!(),
        Some(v) => assert_eq!(*v, 2),
    }
}

#[test]
fn test_eq() {
    let mut m1 = HashMap::new();
    m1.insert(1, 2);
    m1.insert(2, 3);
    m1.insert(3, 4);

    let mut m2 = HashMap::new();
    m2.insert(1, 2);
    m2.insert(2, 3);

    assert!(m1 != m2);

    m2.insert(3, 4);

    assert_eq!(m1, m2);
}

#[test]
fn test_show() {
    let mut map = HashMap::new();
    let empty: HashMap<i32, i32> = HashMap::new();

    map.insert(1, 2);
    map.insert(3, 4);

    let map_str = format!("{:?}", map);

    assert!(map_str == "{1: 2, 3: 4}" || map_str == "{3: 4, 1: 2}");
    assert_eq!(format!("{:?}", empty), "{}");
}

#[test]
fn test_reserve_shrink_to_fit() {
    let mut m = HashMap::new();
    m.insert(0, 0);
    m.remove(&0);
    assert!(m.capacity() >= m.len());
    for i in 0..128 {
        m.insert(i, i);
    }
    m.reserve(256);

    let usable_cap = m.capacity();
    for i in 128..(128 + 256) {
        m.insert(i, i);
        assert_eq!(m.capacity(), usable_cap);
    }

    for i in 100..(128 + 256) {
        assert_eq!(m.remove(&i), Some(i));
    }
    m.shrink_to_fit();

    assert_eq!(m.len(), 100);
    assert!(!m.is_empty());
    assert!(m.capacity() >= m.len());

    for i in 0..100 {
        assert_eq!(m.remove(&i), Some(i));
    }
    m.shrink_to_fit();
    m.insert(0, 0);

    assert_eq!(m.len(), 1);
    assert!(m.capacity() >= m.len());
    assert_eq!(m.remove(&0), Some(0));
}

#[test]
fn test_from_iter() {
    let xs = [(1, 1), (2, 2), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

    let map: HashMap<_, _> = xs.iter().cloned().collect();

    for &(k, v) in &xs {
        assert_eq!(map.get(&k), Some(&v));
    }

    assert_eq!(map.iter().len(), xs.len() - 1);
}

#[test]
fn test_size_hint() {
    let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

    let map: HashMap<_, _> = xs.iter().cloned().collect();

    let mut iter = map.iter();

    for _ in iter.by_ref().take(3) {}

    assert_eq!(iter.size_hint(), (3, Some(3)));
}

#[test]
fn test_iter_len() {
    let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

    let map: HashMap<_, _> = xs.iter().cloned().collect();

    let mut iter = map.iter();

    for _ in iter.by_ref().take(3) {}

    assert_eq!(iter.len(), 3);
}

#[test]
fn test_mut_size_hint() {
    let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

    let mut map: HashMap<_, _> = xs.iter().cloned().collect();

    let mut iter = map.iter_mut();

    for _ in iter.by_ref().take(3) {}

    assert_eq!(iter.size_hint(), (3, Some(3)));
}

#[test]
fn test_iter_mut_len() {
    let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

    let mut map: HashMap<_, _> = xs.iter().cloned().collect();

    let mut iter = map.iter_mut();

    for _ in iter.by_ref().take(3) {}

    assert_eq!(iter.len(), 3);
}

#[test]
fn test_index() {
    let mut map = HashMap::new();

    map.insert(1, 2);
    map.insert(2, 1);
    map.insert(3, 4);

    assert_eq!(map[&2], 1);
}

#[test]
#[should_panic]
fn test_index_nonexistent() {
    let mut map = HashMap::new();

    map.insert(1, 2);
    map.insert(2, 1);
    map.insert(3, 4);

    map[&4];
}

#[test]
fn test_entry() {
    let xs = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

    let mut map: HashMap<_, _> = xs.iter().cloned().collect();

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
            let new_v = (*v) * 10;
            *v = new_v;
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
fn test_entry_take_doesnt_corrupt() {
    #![allow(deprecated)] //rand
    // Test for #19292
    fn check(m: &HashMap<i32, ()>) {
        for k in m.keys() {
            assert!(m.contains_key(k), "{} is in keys() but not in the map?", k);
        }
    }

    let mut m = HashMap::new();
    let mut rng = thread_rng();

    // Populate the map with some items.
    for _ in 0..50 {
        let x = rng.gen_range(-10, 10);
        m.insert(x, ());
    }

    for _ in 0..1000 {
        let x = rng.gen_range(-10, 10);
        match m.entry(x) {
            Vacant(_) => {}
            Occupied(e) => {
                e.remove();
            }
        }

        check(&m);
    }
}

#[test]
fn test_extend_ref() {
    let mut a = HashMap::new();
    a.insert(1, "one");
    let mut b = HashMap::new();
    b.insert(2, "two");
    b.insert(3, "three");

    a.extend(&b);

    assert_eq!(a.len(), 3);
    assert_eq!(a[&1], "one");
    assert_eq!(a[&2], "two");
    assert_eq!(a[&3], "three");
}

#[test]
fn test_capacity_not_less_than_len() {
    let mut a = HashMap::new();
    let mut item = 0;

    for _ in 0..116 {
        a.insert(item, 0);
        item += 1;
    }

    assert!(a.capacity() > a.len());

    let free = a.capacity() - a.len();
    for _ in 0..free {
        a.insert(item, 0);
        item += 1;
    }

    assert_eq!(a.len(), a.capacity());

    // Insert at capacity should cause allocation.
    a.insert(item, 0);
    assert!(a.capacity() > a.len());
}

#[test]
fn test_occupied_entry_key() {
    let mut a = HashMap::new();
    let key = "hello there";
    let value = "value goes here";
    assert!(a.is_empty());
    a.insert(key, value);
    assert_eq!(a.len(), 1);
    assert_eq!(a[key], value);

    match a.entry(key) {
        Vacant(_) => panic!(),
        Occupied(e) => assert_eq!(key, *e.key()),
    }
    assert_eq!(a.len(), 1);
    assert_eq!(a[key], value);
}

#[test]
fn test_vacant_entry_key() {
    let mut a = HashMap::new();
    let key = "hello there";
    let value = "value goes here";

    assert!(a.is_empty());
    match a.entry(key) {
        Occupied(_) => panic!(),
        Vacant(e) => {
            assert_eq!(key, *e.key());
            e.insert(value);
        }
    }
    assert_eq!(a.len(), 1);
    assert_eq!(a[key], value);
}

#[test]
fn test_retain() {
    let mut map: HashMap<i32, i32> = (0..100).map(|x| (x, x * 10)).collect();

    map.retain(|&k, _| k % 2 == 0);
    assert_eq!(map.len(), 50);
    assert_eq!(map[&2], 20);
    assert_eq!(map[&4], 40);
    assert_eq!(map[&6], 60);
}

#[test]
#[cfg_attr(target_os = "android", ignore)] // Android used in CI has a broken dlmalloc
fn test_try_reserve() {
    let mut empty_bytes: HashMap<u8, u8> = HashMap::new();

    const MAX_USIZE: usize = usize::MAX;

    assert_matches!(
        empty_bytes.try_reserve(MAX_USIZE).map_err(|e| e.kind()),
        Err(CapacityOverflow),
        "usize::MAX should trigger an overflow!"
    );

    if let Err(AllocError { .. }) = empty_bytes.try_reserve(MAX_USIZE / 16).map_err(|e| e.kind()) {
    } else {
        // This may succeed if there is enough free memory. Attempt to
        // allocate a few more hashmaps to ensure the allocation will fail.
        let mut empty_bytes2: HashMap<u8, u8> = HashMap::new();
        let _ = empty_bytes2.try_reserve(MAX_USIZE / 16);
        let mut empty_bytes3: HashMap<u8, u8> = HashMap::new();
        let _ = empty_bytes3.try_reserve(MAX_USIZE / 16);
        let mut empty_bytes4: HashMap<u8, u8> = HashMap::new();
        assert_matches!(
            empty_bytes4.try_reserve(MAX_USIZE / 16).map_err(|e| e.kind()),
            Err(AllocError { .. }),
            "usize::MAX / 16 should trigger an OOM!"
        );
    }
}

#[test]
fn test_raw_entry() {
    use super::RawEntryMut::{Occupied, Vacant};

    let xs = [(1i32, 10i32), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

    let mut map: HashMap<_, _> = xs.iter().cloned().collect();

    let compute_hash = |map: &HashMap<i32, i32>, k: i32| -> u64 {
        use core::hash::{BuildHasher, Hash, Hasher};

        let mut hasher = map.hasher().build_hasher();
        k.hash(&mut hasher);
        hasher.finish()
    };

    // Existing key (insert)
    match map.raw_entry_mut().from_key(&1) {
        Vacant(_) => unreachable!(),
        Occupied(mut view) => {
            assert_eq!(view.get(), &10);
            assert_eq!(view.insert(100), 10);
        }
    }
    let hash1 = compute_hash(&map, 1);
    assert_eq!(map.raw_entry().from_key(&1).unwrap(), (&1, &100));
    assert_eq!(map.raw_entry().from_hash(hash1, |k| *k == 1).unwrap(), (&1, &100));
    assert_eq!(map.raw_entry().from_key_hashed_nocheck(hash1, &1).unwrap(), (&1, &100));
    assert_eq!(map.len(), 6);

    // Existing key (update)
    match map.raw_entry_mut().from_key(&2) {
        Vacant(_) => unreachable!(),
        Occupied(mut view) => {
            let v = view.get_mut();
            let new_v = (*v) * 10;
            *v = new_v;
        }
    }
    let hash2 = compute_hash(&map, 2);
    assert_eq!(map.raw_entry().from_key(&2).unwrap(), (&2, &200));
    assert_eq!(map.raw_entry().from_hash(hash2, |k| *k == 2).unwrap(), (&2, &200));
    assert_eq!(map.raw_entry().from_key_hashed_nocheck(hash2, &2).unwrap(), (&2, &200));
    assert_eq!(map.len(), 6);

    // Existing key (take)
    let hash3 = compute_hash(&map, 3);
    match map.raw_entry_mut().from_key_hashed_nocheck(hash3, &3) {
        Vacant(_) => unreachable!(),
        Occupied(view) => {
            assert_eq!(view.remove_entry(), (3, 30));
        }
    }
    assert_eq!(map.raw_entry().from_key(&3), None);
    assert_eq!(map.raw_entry().from_hash(hash3, |k| *k == 3), None);
    assert_eq!(map.raw_entry().from_key_hashed_nocheck(hash3, &3), None);
    assert_eq!(map.len(), 5);

    // Nonexistent key (insert)
    match map.raw_entry_mut().from_key(&10) {
        Occupied(_) => unreachable!(),
        Vacant(view) => {
            assert_eq!(view.insert(10, 1000), (&mut 10, &mut 1000));
        }
    }
    assert_eq!(map.raw_entry().from_key(&10).unwrap(), (&10, &1000));
    assert_eq!(map.len(), 6);

    // Ensure all lookup methods produce equivalent results.
    for k in 0..12 {
        let hash = compute_hash(&map, k);
        let v = map.get(&k).cloned();
        let kv = v.as_ref().map(|v| (&k, v));

        assert_eq!(map.raw_entry().from_key(&k), kv);
        assert_eq!(map.raw_entry().from_hash(hash, |q| *q == k), kv);
        assert_eq!(map.raw_entry().from_key_hashed_nocheck(hash, &k), kv);

        match map.raw_entry_mut().from_key(&k) {
            Occupied(mut o) => assert_eq!(Some(o.get_key_value()), kv),
            Vacant(_) => assert_eq!(v, None),
        }
        match map.raw_entry_mut().from_key_hashed_nocheck(hash, &k) {
            Occupied(mut o) => assert_eq!(Some(o.get_key_value()), kv),
            Vacant(_) => assert_eq!(v, None),
        }
        match map.raw_entry_mut().from_hash(hash, |q| *q == k) {
            Occupied(mut o) => assert_eq!(Some(o.get_key_value()), kv),
            Vacant(_) => assert_eq!(v, None),
        }
    }
}

mod test_drain_filter {
    use super::*;

    use crate::panic::{catch_unwind, AssertUnwindSafe};
    use crate::sync::atomic::{AtomicUsize, Ordering};

    trait EqSorted: Iterator {
        fn eq_sorted<I: IntoIterator<Item = Self::Item>>(self, other: I) -> bool;
    }

    impl<T: Iterator> EqSorted for T
    where
        T::Item: Eq + Ord,
    {
        fn eq_sorted<I: IntoIterator<Item = Self::Item>>(self, other: I) -> bool {
            let mut v: Vec<_> = self.collect();
            v.sort_unstable();
            v.into_iter().eq(other)
        }
    }

    #[test]
    fn empty() {
        let mut map: HashMap<i32, i32> = HashMap::new();
        map.drain_filter(|_, _| unreachable!("there's nothing to decide on"));
        assert!(map.is_empty());
    }

    #[test]
    fn consuming_nothing() {
        let pairs = (0..3).map(|i| (i, i));
        let mut map: HashMap<_, _> = pairs.collect();
        assert!(map.drain_filter(|_, _| false).eq_sorted(crate::iter::empty()));
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn consuming_all() {
        let pairs = (0..3).map(|i| (i, i));
        let mut map: HashMap<_, _> = pairs.clone().collect();
        assert!(map.drain_filter(|_, _| true).eq_sorted(pairs));
        assert!(map.is_empty());
    }

    #[test]
    fn mutating_and_keeping() {
        let pairs = (0..3).map(|i| (i, i));
        let mut map: HashMap<_, _> = pairs.collect();
        assert!(
            map.drain_filter(|_, v| {
                *v += 6;
                false
            })
            .eq_sorted(crate::iter::empty())
        );
        assert!(map.keys().copied().eq_sorted(0..3));
        assert!(map.values().copied().eq_sorted(6..9));
    }

    #[test]
    fn mutating_and_removing() {
        let pairs = (0..3).map(|i| (i, i));
        let mut map: HashMap<_, _> = pairs.collect();
        assert!(
            map.drain_filter(|_, v| {
                *v += 6;
                true
            })
            .eq_sorted((0..3).map(|i| (i, i + 6)))
        );
        assert!(map.is_empty());
    }

    #[test]
    fn drop_panic_leak() {
        static PREDS: AtomicUsize = AtomicUsize::new(0);
        static DROPS: AtomicUsize = AtomicUsize::new(0);

        struct D;
        impl Drop for D {
            fn drop(&mut self) {
                if DROPS.fetch_add(1, Ordering::SeqCst) == 1 {
                    panic!("panic in `drop`");
                }
            }
        }

        let mut map = (0..3).map(|i| (i, D)).collect::<HashMap<_, _>>();

        catch_unwind(move || {
            drop(map.drain_filter(|_, _| {
                PREDS.fetch_add(1, Ordering::SeqCst);
                true
            }))
        })
        .unwrap_err();

        assert_eq!(PREDS.load(Ordering::SeqCst), 3);
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn pred_panic_leak() {
        static PREDS: AtomicUsize = AtomicUsize::new(0);
        static DROPS: AtomicUsize = AtomicUsize::new(0);

        struct D;
        impl Drop for D {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }

        let mut map = (0..3).map(|i| (i, D)).collect::<HashMap<_, _>>();

        catch_unwind(AssertUnwindSafe(|| {
            drop(map.drain_filter(|_, _| match PREDS.fetch_add(1, Ordering::SeqCst) {
                0 => true,
                _ => panic!(),
            }))
        }))
        .unwrap_err();

        assert_eq!(PREDS.load(Ordering::SeqCst), 2);
        assert_eq!(DROPS.load(Ordering::SeqCst), 1);
        assert_eq!(map.len(), 2);
    }

    // Same as above, but attempt to use the iterator again after the panic in the predicate
    #[test]
    fn pred_panic_reuse() {
        static PREDS: AtomicUsize = AtomicUsize::new(0);
        static DROPS: AtomicUsize = AtomicUsize::new(0);

        struct D;
        impl Drop for D {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }

        let mut map = (0..3).map(|i| (i, D)).collect::<HashMap<_, _>>();

        {
            let mut it = map.drain_filter(|_, _| match PREDS.fetch_add(1, Ordering::SeqCst) {
                0 => true,
                _ => panic!(),
            });
            catch_unwind(AssertUnwindSafe(|| while it.next().is_some() {})).unwrap_err();
            // Iterator behaviour after a panic is explicitly unspecified,
            // so this is just the current implementation:
            let result = catch_unwind(AssertUnwindSafe(|| it.next()));
            assert!(result.is_err());
        }

        assert_eq!(PREDS.load(Ordering::SeqCst), 3);
        assert_eq!(DROPS.load(Ordering::SeqCst), 1);
        assert_eq!(map.len(), 2);
    }
}

#[test]
fn from_array() {
    let map = HashMap::from([(1, 2), (3, 4)]);
    let unordered_duplicates = HashMap::from([(3, 4), (1, 2), (1, 2)]);
    assert_eq!(map, unordered_duplicates);

    // This next line must infer the hasher type parameter.
    // If you make a change that causes this line to no longer infer,
    // that's a problem!
    let _must_not_require_type_annotation = HashMap::from([(1, 2)]);
}
