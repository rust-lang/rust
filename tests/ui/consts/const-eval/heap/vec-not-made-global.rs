#![feature(const_heap)]
const V: Vec<i32> = Vec::with_capacity(1);
//~^ ERROR: encountered `const_allocate` pointer in final value that was not made global

fn main() {}
