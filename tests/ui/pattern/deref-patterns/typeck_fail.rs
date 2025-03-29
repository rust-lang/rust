#![feature(deref_patterns)]
#![allow(incomplete_features)]

fn main() {
    // FIXME(deref_patterns): fails to typecheck because `"foo"` has type &str but deref creates a
    // place of type `str`.
    match "foo".to_string() {
        deref!("foo") => {}
        //~^ ERROR: mismatched types
        "foo" => {}
        //~^ ERROR: mismatched types
        _ => {}
    }
    match &"foo".to_string() {
        deref!("foo") => {}
        //~^ ERROR: mismatched types
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
