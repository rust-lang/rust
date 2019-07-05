// check-pass

#![feature(existential_type)]

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


existential type TE: MyTrait;

fn main() {}
