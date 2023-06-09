#![feature(associated_type_defaults)]

pub trait Foo<T: Default + ToString> {
    type Out: Default + ToString = T;
}

impl Foo<u32> for () {
}

impl Foo<u64> for () {
    type Out = bool;
}
