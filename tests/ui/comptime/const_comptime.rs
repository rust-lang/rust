#![feature(rustc_attrs)]

#[rustc_comptime]
fn foo() {}
//~^ ERROR a function cannot just be `comptime`

fn main() {}
