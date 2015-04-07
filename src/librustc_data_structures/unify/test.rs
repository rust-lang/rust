#![allow(non_snake_case)]

extern crate test;
use self::test::Bencher;
use std::collections::HashSet;
use unify::{UnifyKey, UnificationTable};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
struct UnitKey(u32);

impl UnifyKey for UnitKey {
    type Value = ();
    fn index(&self) -> u32 { self.0 }
    fn from_index(u: u32) -> UnitKey { UnitKey(u) }
    fn tag(_: Option<UnitKey>) -> &'static str { "UnitKey" }
}

#[test]
fn basic() {
    let mut ut: UnificationTable<UnitKey> = UnificationTable::new();
    let k1 = ut.new_key(());
    let k2 = ut.new_key(());
    assert_eq!(ut.unioned(k1, k2), false);
    ut.union(k1, k2);
    assert_eq!(ut.unioned(k1, k2), true);
}

#[test]
fn big_array() {
    let mut ut: UnificationTable<UnitKey> = UnificationTable::new();
    let mut keys = Vec::new();
    const MAX: usize = 1 << 15;

    for _ in 0..MAX {
        keys.push(ut.new_key(()));
    }

    for i in 1..MAX {
        let l = keys[i-1];
        let r = keys[i];
        ut.union(l, r);
    }

    for i in 0..MAX {
        assert!(ut.unioned(keys[0], keys[i]));
    }
}

#[bench]
fn big_array_bench(b: &mut Bencher) {
    let mut ut: UnificationTable<UnitKey> = UnificationTable::new();
    let mut keys = Vec::new();
    const MAX: usize = 1 << 15;

    for _ in 0..MAX {
        keys.push(ut.new_key(()));
    }


    b.iter(|| {
        for i in 1..MAX {
            let l = keys[i-1];
            let r = keys[i];
            ut.union(l, r);
        }

        for i in 0..MAX {
            assert!(ut.unioned(keys[0], keys[i]));
        }
    })
}

#[test]
fn even_odd() {
    let mut ut: UnificationTable<UnitKey> = UnificationTable::new();
    let mut keys = Vec::new();
    const MAX: usize = 1 << 10;

    for i in 0..MAX {
        let key = ut.new_key(());
        keys.push(key);

        if i >= 2 {
            ut.union(key, keys[i-2]);
        }
    }

    for i in 1..MAX {
        assert!(!ut.unioned(keys[i-1], keys[i]));
    }

    for i in 2..MAX {
        assert!(ut.unioned(keys[i-2], keys[i]));
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
struct IntKey(u32);

impl UnifyKey for IntKey {
    type Value = Option<i32>;
    fn index(&self) -> u32 { self.0 }
    fn from_index(u: u32) -> IntKey { IntKey(u) }
    fn tag(_: Option<IntKey>) -> &'static str { "IntKey" }
}

/// Test unifying a key whose value is `Some(_)`  with a key whose value is `None`.
/// Afterwards both should be `Some(_)`.
#[test]
fn unify_key_Some_key_None() {
    let mut ut: UnificationTable<IntKey> = UnificationTable::new();
    let k1 = ut.new_key(Some(22));
    let k2 = ut.new_key(None);
    assert!(ut.unify_var_var(k1, k2).is_ok());
    assert_eq!(ut.probe(k2), Some(22));
    assert_eq!(ut.probe(k1), Some(22));
}

/// Test unifying a key whose value is `None`  with a key whose value is `Some(_)`.
/// Afterwards both should be `Some(_)`.
#[test]
fn unify_key_None_key_Some() {
    let mut ut: UnificationTable<IntKey> = UnificationTable::new();
    let k1 = ut.new_key(Some(22));
    let k2 = ut.new_key(None);
    assert!(ut.unify_var_var(k2, k1).is_ok());
    assert_eq!(ut.probe(k2), Some(22));
    assert_eq!(ut.probe(k1), Some(22));
}

/// Test unifying a key whose value is `Some(x)` with a key whose value is `Some(y)`.
/// This should yield an error.
#[test]
fn unify_key_Some_x_key_Some_y() {
    let mut ut: UnificationTable<IntKey> = UnificationTable::new();
    let k1 = ut.new_key(Some(22));
    let k2 = ut.new_key(Some(23));
    assert_eq!(ut.unify_var_var(k1, k2), Err((22, 23)));
    assert_eq!(ut.unify_var_var(k2, k1), Err((23, 22)));
    assert_eq!(ut.probe(k1), Some(22));
    assert_eq!(ut.probe(k2), Some(23));
}

/// Test unifying a key whose value is `Some(x)` with a key whose value is `Some(x)`.
/// This should be ok.
#[test]
fn unify_key_Some_x_key_Some_x() {
    let mut ut: UnificationTable<IntKey> = UnificationTable::new();
    let k1 = ut.new_key(Some(22));
    let k2 = ut.new_key(Some(22));
    assert!(ut.unify_var_var(k1, k2).is_ok());
    assert_eq!(ut.probe(k1), Some(22));
    assert_eq!(ut.probe(k2), Some(22));
}

/// Test unifying a key whose value is `None` with a value is `x`.
/// Afterwards key should be `x`.
#[test]
fn unify_key_None_val() {
    let mut ut: UnificationTable<IntKey> = UnificationTable::new();
    let k1 = ut.new_key(None);
    assert!(ut.unify_var_value(k1, 22).is_ok());
    assert_eq!(ut.probe(k1), Some(22));
}

/// Test unifying a key whose value is `Some(x)` with the value `y`.
/// This should yield an error.
#[test]
fn unify_key_Some_x_val_y() {
    let mut ut: UnificationTable<IntKey> = UnificationTable::new();
    let k1 = ut.new_key(Some(22));
    assert_eq!(ut.unify_var_value(k1, 23), Err((22, 23)));
    assert_eq!(ut.probe(k1), Some(22));
}

/// Test unifying a key whose value is `Some(x)` with the value `x`.
/// This should be ok.
#[test]
fn unify_key_Some_x_val_x() {
    let mut ut: UnificationTable<IntKey> = UnificationTable::new();
    let k1 = ut.new_key(Some(22));
    assert!(ut.unify_var_value(k1, 22).is_ok());
    assert_eq!(ut.probe(k1), Some(22));
}

