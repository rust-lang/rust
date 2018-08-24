// min-lldb-version: 310

// aux-build:issue13213aux.rs

extern crate issue13213aux;

// compile-flags:-g

// This tests make sure that we get no linker error when using a completely inlined static. Some
// statics that are marked with AvailableExternallyLinkage in the importing crate, may actually not
// be available because they have been optimized out from the exporting crate.
fn main() {
    let b: issue13213aux::S = issue13213aux::A;
    println!("Nothing to do here...");
}
