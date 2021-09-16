#![feature(type_alias_impl_trait)]

fn main() {}

trait TraitWithAssoc {
    type Assoc;
}

type Foo<V> = impl Trait<V>;
//~^ ERROR could not find defining uses

trait Trait<U> {}

impl<W> Trait<W> for () {}

fn foo_desugared<T: TraitWithAssoc>(_: T) -> Foo<T::Assoc> {
    //~^ ERROR non-defining opaque type use in defining scope
    //~| ERROR non-defining opaque type use in defining scope
    //~| ERROR non-defining opaque type use in defining scope
    //~| ERROR `T` is part of concrete type but not used in parameter list
    //~| ERROR `T` is part of concrete type but not used in parameter list
    ()
}
