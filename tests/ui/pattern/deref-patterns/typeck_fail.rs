#![feature(deref_patterns)]
#![allow(incomplete_features)]

fn main() {
    // FIXME(deref_patterns): fails to typecheck because string literal patterns don't peel
    // references from the scrutinee.
    match "foo".to_string() {
        "foo" => {}
        //~^ ERROR: mismatched types
        _ => {}
    }
    match &"foo".to_string() {
        "foo" => {}
        //~^ ERROR: mismatched types
        _ => {}
    }

    // Make sure we don't try implicitly dereferncing any ADT.
    match Some(0) {
        Ok(0) => {}
        //~^ ERROR: mismatched types
        _ => {}
    }
}
