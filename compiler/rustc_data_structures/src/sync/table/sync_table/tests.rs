#![cfg(test)]

use super::SyncTable;
use crate::collect::Pin;
use crate::collect::pin;
use crate::collect::release;
use std::collections::hash_map::RandomState;
use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    hash::Hasher,
};

#[test]
fn high_align() {
    #[repr(align(64))]
    #[derive(Clone, Eq, PartialEq, Hash)]
    struct A(u64);

    let mut table = SyncTable::new();

    table.get_mut(&A(1), None);

    table.lock().insert_new(A(1), 1, None);

    release();
}

#[test]
fn test_create_capacity_zero() {
    let m = SyncTable::new_with(RandomState::new(), 0);

    assert!(m.lock().insert(1, 1, None));

    assert!(m.lock().read().get(&1, None).is_some());
    assert!(m.lock().read().get(&0, None).is_none());

    release();
}

#[test]
fn test_replace() {
    let m = SyncTable::new();
    m.lock().insert(2, 7, None);
    m.lock().insert(5, 3, None);
    m.lock().replace(vec![(3, 4)], 0);
    assert_eq!(*m.lock().read().get(&3, None).unwrap().1, 4);
    assert_eq!(m.lock().read().get(&2, None), None);
    assert_eq!(m.lock().read().get(&5, None), None);
    m.lock().replace(vec![], 0);
    assert_eq!(m.lock().read().get(&3, None), None);
    assert_eq!(m.lock().read().get(&2, None), None);
    assert_eq!(m.lock().read().get(&5, None), None);
    release();
}

#[test]
fn test_remove() {
    let m = SyncTable::new();
    m.lock().insert(2, 7, None);
    m.lock().insert(5, 3, None);
    m.lock().remove(&2, None);
    m.lock().remove(&5, None);
    assert_eq!(m.lock().read().get(&2, None), None);
    assert_eq!(m.lock().read().get(&5, None), None);
    assert_eq!(m.lock().read().len(), 0);
    release();
}

#[test]
fn test_insert() {
    let m = SyncTable::new();
    assert_eq!(m.lock().read().len(), 0);
    assert!(m.lock().insert(1, 2, None));
    assert_eq!(m.lock().read().len(), 1);
    assert!(m.lock().insert(2, 4, None));
    assert_eq!(m.lock().read().len(), 2);
    assert_eq!(*m.lock().read().get(&1, None).unwrap().1, 2);
    assert_eq!(*m.lock().read().get(&2, None).unwrap().1, 4);

    release();
}

#[test]
fn test_iter() {
    let m = SyncTable::new();
    assert!(m.lock().insert(1, 2, None));
    assert!(m.lock().insert(5, 3, None));
    assert!(m.lock().insert(2, 4, None));
    assert!(m.lock().insert(9, 4, None));

    pin(|pin| {
        let mut v: Vec<(i32, i32)> = m.read(pin).iter().map(|i| (*i.0, *i.1)).collect();
        v.sort_by_key(|k| k.0);

        assert_eq!(v, vec![(1, 2), (2, 4), (5, 3), (9, 4)]);
    });

    release();
}

#[test]
fn test_insert_conflicts() {
    let m = SyncTable::new_with(RandomState::default(), 4);
    assert!(m.lock().insert(1, 2, None));
    assert!(m.lock().insert(5, 3, None));
    assert!(m.lock().insert(9, 4, None));
    assert_eq!(*m.lock().read().get(&9, None).unwrap().1, 4);
    assert_eq!(*m.lock().read().get(&5, None).unwrap().1, 3);
    assert_eq!(*m.lock().read().get(&1, None).unwrap().1, 2);

    release();
}

#[test]
fn test_expand() {
    let m = SyncTable::new();

    assert_eq!(m.lock().read().len(), 0);

    let mut i = 0;
    let old_raw_cap = unsafe { m.current().info().buckets() };
    while old_raw_cap == unsafe { m.current().info().buckets() } {
        m.lock().insert(i, i, None);
        i += 1;
    }

    assert_eq!(m.lock().read().len(), i);

    release();
}

#[test]
fn test_find() {
    let m = SyncTable::new();
    assert!(m.lock().read().get(&1, None).is_none());
    m.lock().insert(1, 2, None);
    match m.lock().read().get(&1, None) {
        None => panic!(),
        Some(v) => assert_eq!(*v.1, 2),
    }

    release();
}

#[test]
fn test_capacity_not_less_than_len() {
    let a: SyncTable<i32, i32> = SyncTable::new();
    let mut item = 0;

    for _ in 0..116 {
        a.lock().insert(item, 0, None);
        item += 1;
    }

    pin(|pin| {
        assert!(a.read(pin).capacity() > a.read(pin).len());

        let free = a.read(pin).capacity() - a.read(pin).len();
        for _ in 0..free {
            a.lock().insert(item, 0, None);
            item += 1;
        }

        assert_eq!(a.read(pin).len(), a.read(pin).capacity());

        // Insert at capacity should cause allocation.
        a.lock().insert(item, 0, None);
        assert!(a.read(pin).capacity() > a.read(pin).len());
    });

    release();
}

#[test]
fn rehash() {
    let table = SyncTable::new();
    for i in 0..100 {
        table.lock().insert_new(i, (), None);
    }

    pin(|pin| {
        for i in 0..100 {
            assert_eq!(table.read(pin).get(&i, None).map(|b| *b.0), Some(i));
            assert!(table.read(pin).get(&(i + 100), None).is_none());
        }
    });

    release();
}

const INTERN_SIZE: u64 = if cfg!(miri) { 35 } else { 26334 };
const HIT_RATE: u64 = 84;

fn assert_equal(a: &mut SyncTable<u64, u64>, b: &HashMap<u64, u64>) {
    let mut ca: Vec<_> = b.iter().map(|v| (*v.0, *v.1)).collect();
    ca.sort();
    let mut cb: Vec<_> = a.write().read().iter().map(|v| (*v.0, *v.1)).collect();
    cb.sort();
    assert_eq!(ca, cb);
}

fn test_interning(intern: impl Fn(&SyncTable<u64, u64>, u64, u64, Pin<'_>) -> bool) {
    let mut control = HashMap::new();
    let mut test = SyncTable::new();

    for i in 0..INTERN_SIZE {
        let mut s = DefaultHasher::new();
        s.write_u64(i);
        let s = s.finish();
        if s % 100 > (100 - HIT_RATE) {
            test.lock().insert(i, i * 2, None);
            control.insert(i, i * 2);
        }
    }

    assert_equal(&mut test, &control);

    pin(|pin| {
        for i in 0..INTERN_SIZE {
            assert_eq!(
                intern(&test, i, i * 2, pin),
                control.insert(i, i * 2).is_some()
            )
        }
    });

    assert_equal(&mut test, &control);

    release();
}

#[test]
fn intern_potential() {
    fn intern(table: &SyncTable<u64, u64>, k: u64, v: u64, pin: Pin<'_>) -> bool {
        let hash = table.hash_key(&k);
        let p = match table.read(pin).get_potential(&k, Some(hash)) {
            Ok(_) => return true,
            Err(p) => p,
        };

        let mut write = table.lock();
        match p.get(write.read(), &k, Some(hash)) {
            Some(v) => {
                v.1;
                true
            }
            None => {
                p.insert_new(&mut write, k, v, Some(hash));
                false
            }
        }
    }

    test_interning(intern);
}

#[test]
fn intern_get_insert() {
    fn intern(table: &SyncTable<u64, u64>, k: u64, v: u64, pin: Pin<'_>) -> bool {
        let hash = table.hash_key(&k);
        match table.read(pin).get(&k, Some(hash)) {
            Some(_) => return true,
            None => (),
        };

        let mut write = table.lock();
        match write.read().get(&k, Some(hash)) {
            Some(_) => true,
            None => {
                write.insert_new(k, v, Some(hash));
                false
            }
        }
    }

    test_interning(intern);
}

#[test]
fn intern_potential_try() {
    fn intern(table: &SyncTable<u64, u64>, k: u64, v: u64, pin: Pin<'_>) -> bool {
        let hash = table.hash_key(&k);
        let p = match table.read(pin).get_potential(&k, Some(hash)) {
            Ok(_) => return true,
            Err(p) => p,
        };

        let mut write = table.lock();

        write.reserve_one();

        let p = p.refresh(table.read(pin), &k, Some(hash));

        match p {
            Ok(_) => true,
            Err(p) => {
                p.try_insert_new(&mut write, k, v, Some(hash)).unwrap();
                false
            }
        }
    }

    test_interning(intern);
}
