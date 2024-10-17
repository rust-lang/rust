// Check that we do not allow casts or coercions
// to dyn-incompatible trait objects by ref

#![feature(dyn_compatible_for_dispatch)]

trait Trait: Sized {}

struct S;

impl Trait for S {}

fn takes_trait(t: &dyn Trait) {}

fn main() {
    &S as &dyn Trait; //~ ERROR E0038
    let t: &dyn Trait = &S; //~ ERROR E0038
    takes_trait(&S); //~ ERROR E0038
}
