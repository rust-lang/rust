#![allow(incomplete_features)]
#![feature(unnamed_fields)]

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Foo {
    a: u8,
}

#[repr(C)]
#[derive(Debug)]  //~ ERROR only `Copy` and `Clone` may be derived on structs with unnamed fields
struct TestUnsupported {
    _: Foo,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Test {
    _: Foo,
}

#[repr(C)]
#[derive(Clone)]  //~ ERROR deriving `Clone` on a type with unnamed fields requires also deriving `Copy`
struct TestClone {
     _: Foo,
}


fn main() {}
