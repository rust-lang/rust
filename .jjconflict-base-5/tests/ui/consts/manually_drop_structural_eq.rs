// Check that `ManuallyDrop` types can be used as a constant when matching.
// I.e. that `ManuallyDrop` implements `StructuralPartialEq`.
//
// Regression test for <https://github.com/rust-lang/rust/issues/154890>.
//
//@ check-pass
use std::mem::ManuallyDrop;

fn main() {
    const X: ManuallyDrop<u32> = ManuallyDrop::new(1);

    match ManuallyDrop::new(1) {
        X => println!("blah"),
        _ => println!("bleh"),
    }
}
