#![feature(type_alias_impl_trait)]

// Regression test for issue #61863

trait MyTrait {}

#[derive(Debug)]
struct MyStruct {
    v: u64,
}

impl MyTrait for MyStruct {}

fn bla() -> TE {
    return MyStruct { v: 1 };
}

fn bla2() -> TE {
    //~^ ERROR: item does not constrain `TE::{opaque#0}`, but has it in its signature
    bla()
}

type TE = impl MyTrait;

fn main() {}
