//@ run-rustfix
#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

fn main() {}

trait TraitWithAssoc {
    type Assoc;
}

type Foo<V> = impl Trait<V::Assoc>; //~ associated type `Assoc` not found for `V`

trait Trait<U> {}

impl<W> Trait<W> for () {}

fn foo_desugared<T: TraitWithAssoc>(_: T) -> Foo<T> {
    ()
}
