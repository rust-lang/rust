struct Foo {
    foo: u8,
    _: union { //~ ERROR unnamed fields are not yet fully implemented [E0658]
    //~^ ERROR unnamed fields are not yet fully implemented [E0658]
    //~| ERROR anonymous unions are unimplemented
        bar: u8,
        baz: u16
    }
}

union Bar {
    foobar: u8,
    _: struct { //~ ERROR unnamed fields are not yet fully implemented [E0658]
    //~^ ERROR unnamed fields are not yet fully implemented [E0658]
    //~| ERROR anonymous structs are unimplemented
    //~| ERROR unions may not contain fields that need dropping [E0740]
        foobaz: u8,
        barbaz: u16
    }
}

struct S;
struct Baz {
    _: S //~ ERROR unnamed fields are not yet fully implemented [E0658]
}

fn main(){}
