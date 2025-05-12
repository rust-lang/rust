//@ compile-flags: -Znext-solver
//@ check-pass

#![feature(rustc_attrs)]

#[rustc_coinductive]
trait Trait {}

#[rustc_coinductive]
trait Indirect {}
impl<T: Trait + ?Sized> Indirect for T {}

impl<'a> Trait for &'a () where &'a (): Indirect {}

fn impls_trait<T: Trait>() {}

fn main() {
    impls_trait::<&'static ()>();
}
