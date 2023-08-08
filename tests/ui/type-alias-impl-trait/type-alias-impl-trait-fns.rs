// check-pass

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
    bla()
}

type TE = impl MyTrait;

fn main() {}
