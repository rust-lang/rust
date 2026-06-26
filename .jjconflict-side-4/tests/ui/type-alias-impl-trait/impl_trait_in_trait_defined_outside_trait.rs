//! Check that we cannot instantiate a hidden type in the body
//! of an assoc fn or const unless mentioned in the signature.

#![feature(impl_trait_in_assoc_type)]

trait Trait: Sized {
    type Assoc;
    fn foo();
    fn bar() -> Self::Assoc;
}

impl Trait for () {
    type Assoc = impl std::fmt::Debug;
    fn foo() {
        let x: Self::Assoc = 42; //~ ERROR: mismatched types
    }
    fn bar() -> Self::Assoc {
        ""
    }
}

trait Trait2: Sized {
    type Assoc;
    const FOO: ();
    fn bar() -> Self::Assoc;
}

impl Trait2 for () {
    type Assoc = impl std::fmt::Debug;
    const FOO: () = {
        let x: Self::Assoc = 42; //~ ERROR: mismatched types
    };
    fn bar() -> Self::Assoc {
        ""
    }
}

fn main() {}
