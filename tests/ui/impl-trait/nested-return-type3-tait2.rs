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

type Sendable = impl Send;
type Traitable = impl Trait<Assoc = Sendable>;
//~^ WARN opaque type `Traitable` does not satisfy its associated type bounds

fn foo() -> Traitable {
    42
}

fn main() {
}
