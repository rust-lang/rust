// check-pass

trait Duh {}

impl Duh for i32 {}

trait Trait {
    type Assoc: Duh;
}

impl<F: Duh> Trait for F {
    type Assoc = F;
}

fn foo() -> impl Trait<Assoc = impl Send> {
    //~^ WARN opaque type `impl Trait<Assoc = impl Send>` does not satisfy its associated type bounds
    42
}

fn main() {
}
