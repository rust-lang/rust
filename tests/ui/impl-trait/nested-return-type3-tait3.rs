//@ check-pass

#![feature(type_alias_impl_trait)]

trait Duh {}

impl Duh for i32 {}

trait Trait {
    type Assoc: Duh;
}

impl<F: Duh> Trait for F {
    type Assoc = F;
}

type Traitable = impl Trait<Assoc = impl Send>;
//~^ WARN opaque type `Traitable` does not satisfy its associated type bounds

#[define_opaque(Traitable)]
fn foo() -> Traitable {
    42
}

fn main() {}
