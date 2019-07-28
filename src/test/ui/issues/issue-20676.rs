// run-pass
// Regression test for #20676. Error was that we didn't support
// UFCS-style calls to a method in `Trait` where `Self` was bound to a
// trait object of type `Trait`. See also `ufcs-trait-object.rs`.


use std::fmt;

fn main() {
    let a: &dyn fmt::Debug = &1;
    format!("{:?}", a);
}
