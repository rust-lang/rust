// check-pass

#![feature(impl_trait_in_assoc_type)]

struct Bar;

trait Trait: Sized {
    type Assoc;
    fn foo() -> Foo<Self>;
}

impl Trait for Bar {
    type Assoc = impl std::fmt::Debug;
    fn foo() -> Foo<Bar> {
        Foo { field: () }
    }
}

struct Foo<T: Trait> {
    field: <T as Trait>::Assoc,
}

fn main() {}
