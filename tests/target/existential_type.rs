// Opaque type.

#![feature(type_alias_impl_trait)]

pub type Adder<F, T>
where
    T: Clone,
    F: Copy,
= Fn(T) -> T;

pub type Adderrr<T> = Fn(T) -> T;

impl Foo for Bar {
    type E = impl Trait;
}
