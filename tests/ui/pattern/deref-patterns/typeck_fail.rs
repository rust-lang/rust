#![feature(deref_patterns)]

fn main() {
    // Make sure we don't try implicitly dereferncing any ADT.
    match Some(0) {
        Ok(0) => {}
        //~^ ERROR: mismatched types
        _ => {}
    }
}
