//! Check that non-defining assoc items can use the opaque type
//! opaquely.

//@ check-pass

#![feature(impl_trait_in_assoc_type)]

trait Trait: Sized {
    type Assoc;
    fn foo();
    fn bar() -> Self::Assoc;
}

impl Trait for () {
    type Assoc = impl std::fmt::Debug;
    fn foo() {
        let x: Self::Assoc = Self::bar();
    }
    fn bar() -> Self::Assoc {
        ""
    }
}

trait Trait2: Sized {
    type Assoc;
    const FOO: ();
    const BAR: Self::Assoc;
}

impl Trait2 for () {
    type Assoc = impl Copy;
    const FOO: () = {
        let x: Self::Assoc = Self::BAR;
    };
    const BAR: Self::Assoc = "";
}

fn main() {}
