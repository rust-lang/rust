//@ compile-flags: -Cmetadata=aux

#![allow(unconstructable_pub_struct)]

pub trait Foo {}

pub struct Bar<T> { x: T }

impl<T> Foo for Bar<[T; 1 + 1 + 1]> {}
