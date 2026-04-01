//@ run-rustfix

#![allow(unused)]

struct Wrapper<T>(T);

fn bar() -> _ { Wrapper(foo) }
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types

fn foo() {}

fn main() {}
