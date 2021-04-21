struct Foo {
    _: struct { //~ ERROR unnamed fields are not yet fully implemented
        foo: u8
    }
}

union Bar {
    _: union { //~ ERROR unnamed fields are not yet fully implemented
        bar: u8
    }
}

struct S;
struct Baz {
    _: S //~ ERROR unnamed fields are not yet fully implemented
}
