//! Tests the interaction of associated type defaults and specialization.

#![feature(associated_type_defaults, specialization)]
//~^ WARN the feature `specialization` is incomplete

trait Tr {
    type Ty = u8;

    fn make() -> Self::Ty {
        0u8
        //~^ error: mismatched types
    }
}

struct A<T>(T);
// In a `default impl`, assoc. types are defaulted as well,
// so their values can't be assumed.
default impl<T> Tr for A<T> {
    fn make() -> u8 { 0 }
    //~^ ERROR method `make` has an incompatible type for trait
}

struct A2<T>(T);
// ...same, but in the method body
default impl<T> Tr for A2<T> {
    fn make() -> Self::Ty { 0u8 }
    //~^ ERROR mismatched types
}

struct B<T>(T);
// Explicitly defaulting the type does the same.
impl<T> Tr for B<T> {
    default type Ty = bool;

    fn make() -> bool { true }
    //~^ ERROR method `make` has an incompatible type for trait
}

struct B2<T>(T);
// ...same, but in the method body
impl<T> Tr for B2<T> {
    default type Ty = bool;

    fn make() -> Self::Ty { true }
    //~^ ERROR mismatched types
}

struct C<T>(T);
// Only the method is defaulted, so this is fine.
impl<T> Tr for C<T> {
    type Ty = bool;

    default fn make() -> bool { true }
}

// Defaulted method *can* assume the type, if the default is kept.
struct D<T>(T);
impl<T> Tr for D<T> {
    default fn make() -> u8 { 0 }
}

impl Tr for D<bool> {
    fn make() -> u8 { 255 }
}

struct E<T>(T);
impl<T> Tr for E<T> {
    default type Ty = bool;
    default fn make() -> Self::Ty { panic!(); }
}

// This impl specializes and sets `Ty`, it can rely on `Ty=String`.
impl Tr for E<bool> {
    type Ty = String;

    fn make() -> String { String::new() }
}

fn main() {
    // Test that we can assume the right set of assoc. types from outside the impl

    // This is a `default impl`, which does *not* mean that `A`/`A2` actually implement the trait.
    // cf. https://github.com/rust-lang/rust/issues/48515
    //let _: <A<()> as Tr>::Ty = 0u8;
    //let _: <A2<()> as Tr>::Ty = 0u8;

    let _: <B<()> as Tr>::Ty = 0u8;   //~ error: mismatched types
    let _: <B<()> as Tr>::Ty = true;  //~ error: mismatched types
    let _: <B2<()> as Tr>::Ty = 0u8;  //~ error: mismatched types
    let _: <B2<()> as Tr>::Ty = true; //~ error: mismatched types

    let _: <C<()> as Tr>::Ty = true;

    let _: <D<()> as Tr>::Ty = 0u8;
    let _: <D<bool> as Tr>::Ty = 0u8;
}
