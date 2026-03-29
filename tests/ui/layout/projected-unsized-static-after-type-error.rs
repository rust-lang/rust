//@ edition: 2021

// Regression test for error recovery creating a static with an unsized recovered type.
// This used to ICE when CTFE tried to dereference `&X` while validating `Y`.

struct Thing(<&[fn()] as ::core::ops::Deref>::Target);
//~^ ERROR missing lifetime specifier

static X: Thing = Thing(&X);

const Y: &Thing = &X;
//~^ ERROR the type `Thing` has an unknown layout

fn main() {
    let _ = Y;
}
