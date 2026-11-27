#![feature(supertrait_item_shadowing)]
#![allow(dead_code)]

trait Base {
    type Assoc;
}

trait Sub<T>: Base {
    type Assoc;
}

trait Child: Sub<u8> + Sub<u16> {}

// Shadowing Base::Assoc does not resolve the ambiguity between the two Sub bounds.
fn forward<T: Sub<u8> + Sub<u16>>(_: T::Assoc) {}
//~^ ERROR ambiguous associated type `Assoc` in bounds of `T`
fn reversed<T: Sub<u16> + Sub<u8>>(_: T::Assoc) {}
//~^ ERROR ambiguous associated type `Assoc` in bounds of `T`
fn inherited<T: Child<Assoc = ()>>() {}
//~^ ERROR ambiguous associated type `Assoc` in bounds of `Child`

// Same, but without shadowing. This was previously incorrectly accepted when
// supertrait_item_shadowing was enabled, even though no shadowing occurred.
trait Plain<T> {
    type Assoc;
}
fn plain<T: Plain<u8> + Plain<u16>>(_: T::Assoc) {}
//~^ ERROR ambiguous associated type `Assoc` in bounds of `T`

fn main() {}
