mod foo {
    struct A;
    mod bar {
        struct B;
    }
}

struct Foo {
    a: foo:A,
    //~^ ERROR found single colon in a struct field type path
    //~| ERROR expected `,`, or `}`, found `:`
}

struct Bar {
    b: foo::bar:B,
    //~^ ERROR found single colon in a struct field type path
    //~| ERROR expected `,`, or `}`, found `:`
}

fn main() {}
