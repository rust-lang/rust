#![feature(specialization)]
#![allow(incomplete_features)]

struct MyStruct {}

trait MyTrait {
    type MyType: Default;
}

impl MyTrait for i32 {
    default type MyType = MyStruct;
    //~^ ERROR: the trait bound `MyStruct: Default` is not satisfied
}

fn main() {
    let _x: <i32 as MyTrait>::MyType = <i32 as MyTrait>::MyType::default();
}
