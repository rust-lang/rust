// min-lldb-version: 310

// aux-build:issue-13213-aux.rs

extern crate issue_13213_aux;

// compile-flags:-g

// This tests make sure that we get no linker error when using a completely inlined static. Some
// statics that are marked with AvailableExternallyLinkage in the importing crate, may actually not
// be available because they have been optimized out from the exporting crate.
fn main() {
    let b: issue_13213_aux::S = issue_13213_aux::A;
    println!("Nothing to do here...");
}
