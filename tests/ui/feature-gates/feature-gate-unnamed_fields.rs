#[repr(C)]
struct Foo {
    foo: u8,
    _: union { //~ ERROR unnamed fields are not yet fully implemented [E0658]
    //~^ ERROR unnamed fields are not yet fully implemented [E0658]
        bar: u8,
        baz: u16
    }
}

#[repr(C)]
union Bar {
    foobar: u8,
    _: struct { //~ ERROR unnamed fields are not yet fully implemented [E0658]
    //~^ ERROR unnamed fields are not yet fully implemented [E0658]
        foobaz: u8,
        barbaz: u16
    }
}

#[repr(C)]
struct S;

#[repr(C)]
struct Baz {
    _: S //~ ERROR unnamed fields are not yet fully implemented [E0658]
}

fn main(){}
