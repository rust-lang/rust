// https://github.com/rust-lang/rust/issues/21946
trait Foo {
    type A;
}

struct FooStruct;

impl Foo for FooStruct {
    type A = <FooStruct as Foo>::A;
    //~^ ERROR overflow evaluating the requirement `<FooStruct as Foo>::A == _`
}

fn main() {}
