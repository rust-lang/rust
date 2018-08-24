#![feature(untagged_unions)]

union Foo<T: ?Sized> {
    value: T,
    //~^ ERROR the size for values of type
}

struct Foo2<T: ?Sized> {
    value: T,
    //~^ ERROR the size for values of type
    t: u32,
}

enum Foo3<T: ?Sized> {
    Value(T),
    //~^ ERROR the size for values of type
}

fn main() {}
