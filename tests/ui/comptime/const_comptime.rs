#![feature(rustc_attrs)]

#[rustc_comptime]
const fn foo() {}
//~^ ERROR a function cannot be both `comptime` and `const`

fn main() {}
