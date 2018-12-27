#![allow(const_err)] // make sure we cannot allow away the errors tested here

union DummyUnion {
    u8: u8,
    bool: bool,
}

#[repr(C)]
#[derive(Copy, Clone)]
enum Enum {
    A,
    B,
    C,
}

#[derive(Copy, Clone)]
union Foo {
    a: bool,
    b: Enum,
}

union Bar {
    foo: Foo,
    u8: u8,
}

// the value is not valid for bools
const BAD_BOOL: bool = unsafe { DummyUnion { u8: 42 }.bool};
//~^ ERROR it is undefined behavior to use this value

// The value is not valid for any union variant, but that's fine
// unions are just a convenient way to transmute bits around
const BAD_UNION: Foo = unsafe { Bar { u8: 42 }.foo };


fn main() {}
