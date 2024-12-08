// Test const functions in the library

pub const MY_VEC: Vec<usize> = Vec::new();

// FIXME(#110395)
// pub const MY_VEC2: Vec<usize> = Default::default();

pub const MY_STRING: String = String::new();

// pub const MY_STRING2: String = Default::default();

// pub const MY_BOXED_SLICE: Box<[usize]> = Default::default();
// pub const MY_BOXED_STR: Box<str> = Default::default();

use std::collections::{BTreeMap, BTreeSet};

pub const MY_BTREEMAP: BTreeMap<usize, usize> = BTreeMap::new();
pub const MAP: &'static BTreeMap<usize, usize> = &MY_BTREEMAP;
pub const MAP_LEN: usize = MAP.len();
pub const MAP_IS_EMPTY: bool = MAP.is_empty();

pub const MY_BTREESET: BTreeSet<usize> = BTreeSet::new();
pub const SET: &'static BTreeSet<usize> = &MY_BTREESET;
pub const SET_LEN: usize = SET.len();
pub const SET_IS_EMPTY: bool = SET.is_empty();

#[test]
fn test_const() {
    assert_eq!(MY_VEC, /* MY_VEC */ vec![]);
    assert_eq!(MY_STRING, /* MY_STRING2 */ String::default());

    // assert_eq!(MY_VEC, *MY_BOXED_SLICE);
    // assert_eq!(MY_STRING, *MY_BOXED_STR);

    assert_eq!(MAP_LEN, 0);
    assert_eq!(SET_LEN, 0);
    assert!(MAP_IS_EMPTY && SET_IS_EMPTY);
}
