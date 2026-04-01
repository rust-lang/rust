#![feature(type_alias_impl_trait)]

//@ check-pass

trait Duh {}

impl Duh for i32 {}

trait Trait {
    type Assoc: Duh;
}

// the fact that `R` is the `::Output` projection on `F` causes
// an intermediate inference var to be generated which is then later
// compared against the actually found `Assoc` type.
impl<R: Duh, F: FnMut() -> R> Trait for F {
    type Assoc = R;
}

type Sendable = impl Send;

// The `Sendable` here is converted to an inference var and then later compared
// against the inference var created, causing the inference var to be set to
// the hidden type of `Sendable` instead of
// the hidden type. We already have obligations registered on the inference
// var to make it uphold the `: Duh` bound on `Trait::Assoc`. The opaque
// type does not implement `Duh`, but if its hidden type does.
#[define_opaque(Sendable)]
fn foo() -> impl Trait<Assoc = Sendable> {
    //~^ WARN opaque type `impl Trait<Assoc = Sendable>` does not satisfy its associated type bounds
    || 42
}

fn main() {}
