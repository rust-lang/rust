// Regression test for issue #116334.
// Don't include hygienic fields from different syntax contexts in
// the list of available or similarly named fields.

#![feature(decl_macro)]

macro compound($Ty:ident) {
    #[derive(Default)]
    struct $Ty {
        field: u32, // field `field` is hygienic
    }
}

macro component($Ty:ident) {
    struct $Ty(u64); // field `0` is hygienic (but still accessible via the constructor)
}

compound! { Compound }
component! { Component }

fn main() {
    let ty = Compound::default();

    let _ = ty.field; //~ ERROR no field `field` on type `Compound`
    let _ = ty.fieeld; //~ ERROR no field `fieeld` on type `Compound`

    let Compound { field } = ty;
    //~^ ERROR struct `Compound` does not have a field named `field`
    //~| ERROR pattern requires `..` due to inaccessible fields
    //~| HELP ignore the inaccessible and unused fields

    let ty = Component(90);

    let _ = ty.0; //~ ERROR no field `0` on type `Component`
}

environment!();

macro environment() {
    struct Crate { field: () }

    // Here, we do want to suggest `field` even though it's hygienic
    // precisely because they come from the same syntax context.
    const CRATE: Crate = Crate { fiel: () };
    //~^ ERROR struct `Crate` has no field named `fiel`
    //~| HELP a field with a similar name exists
}
