//! Make sure that `derive(Clone)` works for simple structs and enums.
//@ run-pass
#![allow(dead_code)]

#[derive(Clone)]
enum SimpleEnum {
    A,
    B(()),
    C,
}

#[derive(Clone)]
enum GenericEnum<T, U> {
    A(T),
    B(T, U),
    C,
}

#[derive(Clone)]
struct TupleStruct((), ());

#[derive(Clone)]
struct GenericStruct<T> {
    foo: (),
    bar: (),
    baz: T,
}

#[derive(Clone)]
struct GenericTupleStruct<T>(T, ());

#[derive(Clone)]
struct ManyPrimitives {
    _int: isize,
    _i8: i8,
    _i16: i16,
    _i32: i32,
    _i64: i64,

    _uint: usize,
    _u8: u8,
    _u16: u16,
    _u32: u32,
    _u64: u64,

    _f32: f32,
    _f64: f64,

    _bool: bool,
    _char: char,
    _nil: (),
}

// Regression test for issue #30244
#[derive(Copy, Clone)]
struct Array {
    arr: [[u8; 256]; 4],
}

pub fn main() {
    let _ = SimpleEnum::A.clone();
    let _ = GenericEnum::A::<isize, isize>(1).clone();
    let _ = GenericStruct { foo: (), bar: (), baz: 1 }.clone();
    let _ = GenericTupleStruct(1, ()).clone();
}
