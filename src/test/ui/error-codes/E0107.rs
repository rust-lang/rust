struct Foo<'a>(&'a str);
struct Buzz<'a, 'b>(&'a str, &'b str);

enum Bar {
    A,
    B,
    C,
}

struct Baz<'a, 'b, 'c> {
    buzz: Buzz<'a>,
    //~^ ERROR this struct takes 2 lifetime arguments but only 1 lifetime argument was supplied
    //~| HELP add missing lifetime argument

    bar: Bar<'a>,
    //~^ ERROR this enum takes 0 lifetime arguments but 1 lifetime argument was supplied
    //~| HELP remove these generics

    foo2: Foo<'a, 'b, 'c>,
    //~^ ERROR this struct takes 1 lifetime argument but 3 lifetime arguments were supplied
    //~| HELP remove these lifetime arguments
}

fn main() {}
