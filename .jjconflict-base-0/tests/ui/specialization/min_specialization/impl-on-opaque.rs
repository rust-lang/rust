// Test that specializing on opaque types is allowed

//@ check-pass

#![feature(min_specialization, type_alias_impl_trait)]

trait SpecTrait<U> {
    fn f();
}

impl<U> SpecTrait<U> for () {
    default fn f() {}
}

type Opaque = impl Tuple;

trait Tuple {}

impl Tuple for () {}

impl SpecTrait<Opaque> for () {
    fn f() {}
}

impl SpecTrait<u32> for () {
    fn f() {}
}

#[define_opaque(Opaque)]
fn foo() -> Opaque {}

fn main() {}
