//@ check-pass

#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

fn main() {}

type PartiallyDefined<T> = impl Sized;

#[define_opaque(PartiallyDefined)]
fn partially_defined<T: std::fmt::Debug>(_: T) -> PartiallyDefined<T> {
    4u32
}

type PartiallyDefined2<T> = impl Sized;

#[define_opaque(PartiallyDefined2)]
fn partially_defined2<T: std::fmt::Debug>(_: T) -> PartiallyDefined2<T> {
    4u32
}

#[define_opaque(PartiallyDefined2)]
fn partially_defined22<T>(_: T) -> PartiallyDefined2<T> {
    4u32
}
