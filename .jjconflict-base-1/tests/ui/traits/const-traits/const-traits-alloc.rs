//@ check-pass
#![feature(const_trait_impl, const_default)]
#![allow(dead_code)]
// alloc::string
const STRING: String = Default::default();
// alloc::vec
const VEC: Vec<()> = Default::default();
// alloc::collections::btree::map::BTreeMap
use std::collections::BTreeMap;
const BTREE: BTreeMap<(), ()> = Default::default();

fn main() {}
