mod foo {
    struct A;
    mod bar {
        struct B;
    }
}

struct Foo {
    a: foo:A,
    //~^ ERROR path separator must be a double colon
    //~| ERROR struct `A` is private
}

struct Bar {
    b: foo::bar:B,
    //~^ ERROR path separator must be a double colon
    //~| ERROR module `bar` is private
}

fn main() {}
