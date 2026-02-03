//@ revisions: adt tait ty_alias assoc
#![feature(pin_ergonomics)]
#![cfg_attr(tait, feature(type_alias_impl_trait))]
#![allow(incomplete_features)]

#[pin_v2]
struct Foo;
struct Bar;

#[cfg(adt)]
mod adt {
    use super::*;

    impl Unpin for Foo {}
    //[adt]~^ ERROR explicit impls for the `Unpin` trait are not permitted for structurally pinned types
    impl Unpin for Bar {} // ok
}

#[cfg(ty_alias)]
mod ty_alias {
    use super::*;

    type Identity<T> = T;

    impl Unpin for Identity<Foo> {}
    //[ty_alias]~^ ERROR explicit impls for the `Unpin` trait are not permitted for structurally pinned types
    impl Unpin for Identity<Bar> {} // ok
}

#[cfg(tait)]
mod tait {
    use super::*;

    trait Identity<T> {}

    impl<T> Identity<T> for T {}

    type FooAlias = impl Identity<Foo>;
    type BarAlias = impl Identity<Bar>;

    #[define_opaque(FooAlias)]
    fn foo_alias() -> FooAlias {
        Foo
    }
    #[define_opaque(BarAlias)]
    fn bar_alias() -> BarAlias {
        Bar
    }

    impl Unpin for FooAlias {}
    //[tait]~^ ERROR only traits defined in the current crate can be implemented for arbitrary types
    impl Unpin for BarAlias {}
    //[tait]~^ ERROR only traits defined in the current crate can be implemented for arbitrary types
}

#[cfg(assoc)]
mod assoc {
    use super::*;

    trait Identity {
        type Assoc;
    }

    impl<T> Identity for T {
        type Assoc = T;
    }

    impl Unpin for <Foo as Identity>::Assoc {}
    //[assoc]~^ ERROR cross-crate traits with a default impl, like `Unpin`, can only be implemented for a struct/enum type, not `<Foo as Identity>::Assoc`
    impl Unpin for <Bar as Identity>::Assoc {}
    //[assoc]~^ ERROR cross-crate traits with a default impl, like `Unpin`, can only be implemented for a struct/enum type, not `<Bar as Identity>::Assoc`
}

fn main() {}
