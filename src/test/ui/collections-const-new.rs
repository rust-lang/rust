// check-pass

// Test several functions can be used for constants
// 1. Vec::new()
// 2. String::new()
// 3. BTreeMap::new()
// 4. BTreeSet::new()

#![feature(const_btree_new)]

const MY_VEC: Vec<usize> = Vec::new();

const MY_STRING: String = String::new();

use std::collections::{BTreeMap, BTreeSet};
const MY_BTREEMAP: BTreeMap<u32, u32> = BTreeMap::new();

const MY_BTREESET: BTreeSet<u32> = BTreeSet::new();

fn main() {}
