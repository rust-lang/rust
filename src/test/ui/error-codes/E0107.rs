struct Foo<'a>(&'a str);
struct Buzz<'a, 'b>(&'a str, &'b str);

enum Bar {
    A,
    B,
    C,
}

struct Baz<'a, 'b, 'c> {
    buzz: Buzz<'a>,
    //~^ ERROR E0107
    //~| expected 2 lifetime arguments
    bar: Bar<'a>,
    //~^ ERROR E0107
    //~| unexpected lifetime argument
    foo2: Foo<'a, 'b, 'c>,
    //~^ ERROR E0107
    //~| unexpected lifetime argument
    //~| unexpected lifetime argument
}

fn main() {}
