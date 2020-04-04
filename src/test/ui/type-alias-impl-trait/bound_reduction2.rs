#![feature(type_alias_impl_trait)]

fn main() {
}

trait TraitWithAssoc {
    type Assoc;
}

type Foo<V> = impl Trait<V>;
//~^ ERROR the trait bound `T: TraitWithAssoc` is not satisfied

trait Trait<U> {}

impl<W> Trait<W> for () {}

fn foo_desugared<T: TraitWithAssoc>(_: T) -> Foo<T::Assoc> {
    ()
}
