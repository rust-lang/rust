//@ check-pass

fn foo() -> impl MyTrait {
    panic!();
    MyStruct
}

struct MyStruct;
trait MyTrait {}

impl MyTrait for MyStruct {}

fn main() {}
