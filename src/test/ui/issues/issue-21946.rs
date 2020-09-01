trait Foo {
    type A;
}

struct FooStruct;

impl Foo for FooStruct {
    //~^ ERROR overflow evaluating the requirement `<FooStruct as Foo>::A == _`
    type A = <FooStruct as Foo>::A;
    //~^ ERROR overflow evaluating the requirement `<FooStruct as Foo>::A == _`
}

fn main() {}
