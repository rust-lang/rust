trait Foo {
    type A;
}

struct FooStruct;

impl Foo for FooStruct {
//~^ ERROR overflow evaluating the requirement `<FooStruct as Foo>::A`
    type A = <FooStruct as Foo>::A;
    //~^ ERROR overflow evaluating the requirement `<FooStruct as Foo>::A`
}

fn main() {}
