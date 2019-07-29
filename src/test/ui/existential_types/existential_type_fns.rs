// check-pass

#![feature(type_alias_impl_trait)]

// Regression test for issue #61863

pub trait MyTrait {}

#[derive(Debug)]
pub struct MyStruct {
  v: u64
}

impl MyTrait for MyStruct {}

pub fn bla() -> TE {
    return MyStruct {v:1}
}

pub fn bla2() -> TE {
    bla()
}


type TE = impl MyTrait;

fn main() {}
