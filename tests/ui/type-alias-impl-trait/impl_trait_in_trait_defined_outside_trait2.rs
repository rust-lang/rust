//! Check that we cannot instantiate a hidden type from another assoc type.

trait Trait: Sized {
    type Assoc;
    type Foo;
    fn foo() -> Self::Assoc;
}

impl Trait for () {
    type Assoc = impl std::fmt::Debug;
    type Foo = [(); {
        let x: Self::Assoc = 42; //~ ERROR: mismatched types
        3
    }];
    fn foo() -> Self::Assoc {
        ""
    }
}

fn main() {}
