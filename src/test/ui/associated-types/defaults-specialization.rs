// compile-fail

#![feature(associated_type_defaults, specialization)]

trait Tr {
    type Ty = u8;

    fn make() -> Self::Ty;
}

struct A<T>(T);
// In a `default impl`, assoc. types are defaulted as well,
// so their values can't be assumed.
default impl<T> Tr for A<T> {
    fn make() -> u8 { 0 }
    //~^ ERROR method `make` has an incompatible type for trait
}

struct B<T>(T);
// Explicitly defaulting the type does the same.
impl<T> Tr for B<T> {
    default type Ty = bool;

    fn make() -> bool { true }
    //~^ ERROR method `make` has an incompatible type for trait
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

fn main() {}
