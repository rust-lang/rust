#![feature(deref_patterns)]
#![allow(incomplete_features)]

struct MyPointer;

impl std::ops::Deref for MyPointer {
    type Target = ();
    fn deref(&self) -> &() {
        &()
    }
}

fn main() {
    // Test that we get a trait error if a user attempts implicit deref pats on their own impls.
    // FIXME(deref_patterns): there should be a special diagnostic for missing `DerefPure`.
    match MyPointer {
        () => {}
        //~^ the trait bound `MyPointer: DerefPure` is not satisfied
        _ => {}
    }
}
