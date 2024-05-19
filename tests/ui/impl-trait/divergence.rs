fn foo() -> impl MyTrait {
    //~^ ERROR undefined opaque type
    panic!();
    MyStruct
}

struct MyStruct;
trait MyTrait {}

impl MyTrait for MyStruct {}

fn main() {}
