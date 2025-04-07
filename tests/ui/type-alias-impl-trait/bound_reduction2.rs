#![feature(type_alias_impl_trait)]

fn main() {}

trait TraitWithAssoc {
    type Assoc;
}

type Foo<V> = impl Trait<V>;

trait Trait<U> {}

impl<W> Trait<W> for () {}

#[define_opaque(Foo)]
fn foo_desugared<T: TraitWithAssoc>(_: T) -> Foo<T::Assoc> {
    //~^ ERROR expected generic type parameter, found `<T as TraitWithAssoc>::Assoc`
    ()
}
