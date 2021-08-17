// Test const functions in the library

use core::cmp::Ordering;

// FIXME remove this struct once we put `K: ?const Ord` on BTreeMap::new.
#[derive(PartialEq, Eq, PartialOrd)]
pub struct MyType;

impl const Ord for MyType {
    fn cmp(&self, _: &Self) -> Ordering {
        Ordering::Equal
    }

    fn max(self, _: Self) -> Self {
        Self
    }

    fn min(self, _: Self) -> Self {
        Self
    }

    fn clamp(self, _: Self, _: Self) -> Self {
        Self
    }
}

pub const MY_VEC: Vec<usize> = Vec::new();
pub const MY_VEC2: Vec<usize> = Default::default();

pub const MY_STRING: String = String::new();
pub const MY_STRING2: String = Default::default();

use std::collections::{BTreeMap, BTreeSet};

pub const MY_BTREEMAP: BTreeMap<MyType, MyType> = BTreeMap::new();
pub const MAP: &'static BTreeMap<MyType, MyType> = &MY_BTREEMAP;
pub const MAP_LEN: usize = MAP.len();
pub const MAP_IS_EMPTY: bool = MAP.is_empty();

pub const MY_BTREESET: BTreeSet<MyType> = BTreeSet::new();
pub const SET: &'static BTreeSet<MyType> = &MY_BTREESET;
pub const SET_LEN: usize = SET.len();
pub const SET_IS_EMPTY: bool = SET.is_empty();

#[test]
fn test_const() {
    assert_eq!(MY_VEC, MY_VEC2);
    assert_eq!(MY_STRING, MY_STRING2);

    assert_eq!(MAP_LEN, 0);
    assert_eq!(SET_LEN, 0);
    assert!(MAP_IS_EMPTY && SET_IS_EMPTY);
}
