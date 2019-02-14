#![feature(existential_type)]

fn main() {
}

trait TraitWithAssoc {
    type Assoc;
}

existential type Foo<V>: Trait<V>;
//~^ ERROR could not find defining uses

trait Trait<U> {}

impl<W> Trait<W> for () {}

fn foo_desugared<T: TraitWithAssoc>(_: T) -> Foo<T::Assoc> { //~ ERROR does not fully define
    ()
}
