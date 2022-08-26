// stderr-per-bitwidth
#![allow(const_err)] // make sure we cannot allow away the errors tested here

#[repr(C)]
union DummyUnion {
    unit: (),
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
#[repr(C)]
union Foo {
    a: bool,
    b: Enum,
}

#[repr(C)]
union Bar {
    foo: Foo,
    u8: u8,
}

// the value is not valid for bools
const BAD_BOOL: bool = unsafe { DummyUnion { u8: 42 }.bool};
//~^ ERROR it is undefined behavior to use this value
const UNINIT_BOOL: bool = unsafe { DummyUnion { unit: () }.bool};
//~^ ERROR evaluation of constant value failed
//~| uninitialized

// The value is not valid for any union variant, but that's fine
// unions are just a convenient way to transmute bits around
const BAD_UNION: Foo = unsafe { Bar { u8: 42 }.foo };


fn main() {}
