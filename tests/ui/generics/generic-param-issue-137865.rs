//@ edition: 2021


trait Foo {
    type Assoc<const N: Self>; //~ ERROR: `Self` is forbidden as the type of a const generic parameter
    fn foo() -> Self::Assoc<3>; //~ ERROR: mismatched types
}

fn main() {}
