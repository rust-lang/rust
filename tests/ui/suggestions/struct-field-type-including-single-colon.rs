mod foo {
    struct A;
    mod bar {
        struct B;
    }
}

struct Foo {
    a: foo:A,
    //~^ ERROR found single colon in a struct field type path
}

struct Bar {
    b: foo::bar:B,
    //~^ ERROR found single colon in a struct field type path
}

// Issue #92685.
struct Qux {
    c: Vec<foo:A>,
    //~^ ERROR: struct takes at least 1 generic argument but 0 generic arguments were supplied
    //~| ERROR: associated item constraints are not allowed here
    //~| ERROR: cannot find trait `A` in this scope
}

fn main() {}
