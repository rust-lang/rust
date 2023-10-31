// build-pass
// aux-build: ../ambiguous-4-extern.rs

extern crate ambiguous_4_extern;

fn main() {
    ambiguous_4_extern::id();
    // `warning_ambiguous` had been lost at metadata.
}
