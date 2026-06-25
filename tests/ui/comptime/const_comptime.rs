#![feature(comptime)]

#[comptime]
const fn foo() {}
//~^ ERROR a function cannot be both `comptime` and `const`

fn main() {}
