//@ compile-flags: -Cmetadata=aux

#![allow(unused_unconstructable_pub_structs)]

pub trait Foo {}

pub struct Bar<T> { x: T }

impl<T> Foo for Bar<[T; 1 + 1 + 1]> {}
