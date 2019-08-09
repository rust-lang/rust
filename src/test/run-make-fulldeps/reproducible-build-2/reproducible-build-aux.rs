#![crate_type="lib"]

pub static STATIC: i32 = 1234;

pub struct Struct<T1, T2> {
    _t1: std::marker::PhantomData<T1>,
    _t2: std::marker::PhantomData<T2>,
}

pub fn regular_fn(_: i32) {}

pub fn generic_fn<T1, T2>() {}

impl<T1, T2> Drop for Struct<T1, T2> {
    fn drop(&mut self) {}
}

pub enum Enum {
    Variant1,
    Variant2(u32),
    Variant3 { x: u32 }
}

pub struct TupleStruct(pub i8, pub i16, pub i32, pub i64);

pub trait Trait<T1, T2> {
    fn foo(&self);
}
