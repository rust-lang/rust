#![feature(negative_impls)]

//@ pp-exact

struct Test;

impl !Send for Test {}

pub fn main() {}
