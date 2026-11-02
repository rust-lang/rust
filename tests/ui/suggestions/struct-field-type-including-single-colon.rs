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
    //~^ ERROR: cannot find trait `A` in this scope
}

fn main() {}
