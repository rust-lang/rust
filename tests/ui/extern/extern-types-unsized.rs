// Make sure extern types are !Sized.

#![feature(extern_types)]

extern "C" {
    type A;
}

struct Foo {
    x: u8,
    tail: A,
}

struct Bar<T: ?Sized> {
    x: u8,
    tail: T,
}

fn assert_sized<T>() {}

fn main() {
    assert_sized::<A>();
    //~^ ERROR the size for values of type

    assert_sized::<Foo>();
    //~^ ERROR the size for values of type

    assert_sized::<Bar<A>>();
    //~^ ERROR the size for values of type

    assert_sized::<Bar<Bar<A>>>();
    //~^ ERROR the size for values of type
}
