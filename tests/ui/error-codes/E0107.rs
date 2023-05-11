struct Foo<'a>(&'a str);
struct Buzz<'a, 'b>(&'a str, &'b str);
struct Qux<'a, T>(&'a T);
struct Quux<T>(T);

enum Bar {
    A,
    B,
    C,
}

struct Baz<'a, 'b, 'c> {
    buzz: Buzz<'a>,
    //~^ ERROR struct takes 2 lifetime arguments
    //~| HELP add missing lifetime argument

    bar: Bar<'a>,
    //~^ ERROR enum takes 0 lifetime arguments
    //~| HELP remove these generics

    foo2: Foo<'a, 'b, 'c>,
    //~^ ERROR struct takes 1 lifetime argument
    //~| HELP remove these lifetime arguments

    qux1: Qux<'a, 'b, i32>,
    //~^ ERROR struct takes 1 lifetime argument
    //~| HELP remove this lifetime argument

    qux2: Qux<'a, i32, 'b>,
    //~^ ERROR struct takes 1 lifetime argument
    //~| HELP remove this lifetime argument

    qux3: Qux<'a, 'b, 'c, i32>,
    //~^ ERROR struct takes 1 lifetime argument
    //~| HELP remove these lifetime arguments

    qux4: Qux<'a, i32, 'b, 'c>,
    //~^ ERROR struct takes 1 lifetime argument
    //~| HELP remove these lifetime arguments

    qux5: Qux<'a, 'b, i32, 'c>,
    //~^ ERROR struct takes 1 lifetime argument
    //~| HELP remove this lifetime argument

    quux: Quux<'a, i32, 'b>,
    //~^ ERROR struct takes 0 lifetime arguments
    //~| HELP remove this lifetime argument
}

pub trait T {
    type A;
    type B;
}

fn trait_bound_generic<I: T<u8, u16>>(_i: I) {
    //~^ ERROR trait takes 0 generic arguments
    //~| HELP replace the generic bounds with the associated types
}

fn main() {}
