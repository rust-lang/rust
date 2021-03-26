// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

fn main() {
}

trait TraitWithAssoc {
    type Assoc;
}

type Foo<V> = impl Trait<V::Assoc>; //~ associated type `Assoc` not found for `V`

trait Trait<U> {}

impl<W> Trait<W> for () {}

fn foo_desugared<T: TraitWithAssoc>(_: T) -> Foo<T> {
    ()
}
