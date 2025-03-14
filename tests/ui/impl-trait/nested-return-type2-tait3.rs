//@ check-pass

#![feature(type_alias_impl_trait)]

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

type Traitable = impl Trait<Assoc = impl Send>;
//~^ WARN opaque type `Traitable` does not satisfy its associated type bounds

// The `impl Send` here is then later compared against the inference var
// created, causing the inference var to be set to `impl Send` instead of
// the hidden type. We already have obligations registered on the inference
// var to make it uphold the `: Duh` bound on `Trait::Assoc`. The opaque
// type does not implement `Duh`, even if its hidden type does. So we error out.
#[define_opaque(Traitable)]
fn foo() -> Traitable {
    || 42
}

fn main() {}
