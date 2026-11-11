#![feature(type_alias_impl_trait)]

pub trait TraitWithAssoc {
    type Assoc;
}

pub type Foo<V> = impl Trait<V::Assoc>;
//~^ ERROR
//~| ERROR

pub trait Trait<U> {}

impl<W> Trait<W> for () {}

pub fn foo_desugared<T: TraitWithAssoc>(_: T) -> Foo<T> {
    ()
}
