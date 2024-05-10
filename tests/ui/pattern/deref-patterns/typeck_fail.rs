#![feature(deref_patterns)]
#![allow(incomplete_features)]

fn main() {
    // FIXME(deref_patterns): fails to typecheck because `"foo"` has type &str but deref creates a
    // place of type `str`.
    match "foo".to_string() {
        deref!("foo") => {}
        //~^ ERROR: mismatched types
        _ => {}
    }
    match &"foo".to_string() {
        deref!("foo") => {}
        //~^ ERROR: mismatched types
        _ => {}
    }
}
