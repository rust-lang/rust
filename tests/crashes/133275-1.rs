//@ known-bug: #133275
#![feature(const_trait_impl)]
#![feature(associated_type_defaults)]

const trait Foo3<T>
where
    Self::Baz: Clone,
{
    type Baz = T;
}

pub fn main() {}
