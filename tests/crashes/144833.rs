//@ known-bug: rust-lang/rust#144833
#![feature(type_alias_impl_trait)]
#![allow(incomplete_features)]

type Opaque = impl Sized;

trait Trait {
    fn foo();
}

impl Trait for Opaque {
    #[define_opaque(Opaque)]
    fn foo() {
        let _: Opaque = || {};
        Opaque::foo();
    }
}

fn main() {
    Opaque::foo();
}
