struct Foo {
    foo: i32;
    //~^ ERROR struct fields are separated by `,`
}

union Bar {
    foo: i32;
    //~^ ERROR union fields are separated by `,`
}

enum Baz {
    Qux { foo: i32; }
    //~^ ERROR struct fields are separated by `,`
}

fn main() {
    let _ = Foo { foo: "" }; //~ ERROR mismatched types
}
