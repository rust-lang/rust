//! Regression test for #69136
//! This test checks that the unknown lifetime `'a` doesn't cause
//! ICEs after emitting the error.

#![feature(type_alias_impl_trait)]

trait SomeTrait {}

impl SomeTrait for () {}

trait WithAssoc<A> {
    type AssocType;
}

impl<T> WithAssoc<T> for () {
    type AssocType = ();
}

type Return<A> = impl WithAssoc<A, AssocType = impl SomeTrait + 'a>;
//~^ ERROR use of undeclared lifetime name `'a`

#[define_opaque(Return)]
fn my_fun<T>() -> Return<T> {}

fn main() {}
