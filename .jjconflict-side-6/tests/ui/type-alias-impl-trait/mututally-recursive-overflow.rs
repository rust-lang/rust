//@ revisions: next old
//@[next] compile-flags: -Znext-solver
//@ edition: 2021
//@ build-fail
//[old]~^^^^ ERROR overflow evaluating the requirement `<() as B>::Assoc == _`
//[next]~^^^^^ ERROR: overflow evaluating the requirement `<() as A>::Assoc::{opaque#0} == _`

#![feature(rustc_attrs)]
#![feature(impl_trait_in_assoc_type)]

#[rustc_coinductive]
trait A {
    type Assoc;

    fn test() -> Self::Assoc;
}

#[rustc_coinductive]
trait B {
    type Assoc;

    fn test() -> Self::Assoc;
}

impl<T: A> B for T {
    type Assoc = impl Sized;

    fn test() -> <Self as B>::Assoc {
        <T as A>::test()
    }
}

fn main() {
    <() as A>::test();
}

impl<T: B> A for T {
    type Assoc = impl Sized;

    fn test() -> <Self as A>::Assoc {
        <T as B>::test()
    }
}
