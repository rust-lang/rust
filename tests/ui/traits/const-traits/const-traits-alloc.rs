//@ run-pass
#![feature(const_trait_impl)]
#![allow(dead_code)]
// alloc::string
const STRING: String = Default::default();
// alloc::vec
const VEC: Vec<()> = Default::default();

fn main() {}
