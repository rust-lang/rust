//@ run-pass
#![feature(const_trait_impl, const_default)]
#![allow(dead_code)]
// alloc::string
const STRING: String = Default::default();
// alloc::vec
const VEC: Vec<()> = Default::default();

fn main() {}
