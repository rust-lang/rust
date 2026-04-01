//@ check-pass

trait Duh {}

impl Duh for i32 {}

trait Trait {
    type Assoc: Duh;
}

impl<F: Duh> Trait for F {
    type Assoc = F;
}

fn foo() -> impl Trait<Assoc = impl Send> {
    42
}

fn main() {
}
