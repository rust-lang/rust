// check-pass

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

fn foo() -> impl Trait<Assoc = Sendable> {
    42
}

fn main() {
}
