//! Test that the place behind a deref pattern is treated as maybe-invalid, and thus empty arms
//! cannot be omitted. This is handled the same as for refs and union fields, so this leaves the
//! bulk of the testing to `tests/ui/pattern/usefulness/empty-types.rs`.
// FIXME(deref_patterns): On stabilization, cases for deref patterns could be worked into that file
// to keep the tests for empty types in one place and test more thoroughly.
#![feature(deref_patterns)]
#![expect(incomplete_features)]
#![deny(unreachable_patterns)]

enum Void {}

fn main() {
    // Sanity check: matching on an empty type without pointer indirection lets us omit arms.
    let opt_void: Option<Void> = None;
    match opt_void {
        None => {}
    }

    // But if we hide it behind a smart pointer, we need an arm.
    let box_opt_void: Box<Option<Void>> = Box::new(None);
    match box_opt_void {
        //~^ ERROR non-exhaustive patterns: `deref!(Some(_))` not covered
        None => {}
    }
    match box_opt_void {
        None => {}
        Some(_) => {}
    }
    match box_opt_void {
        None => {}
        _ => {}
    }

    // For consistency, this behaves the same as if we manually dereferenced the scrutinee.
    match *box_opt_void {
        //~^ ERROR non-exhaustive patterns: `Some(_)` not covered
        None => {}
    }
    match *box_opt_void {
        None => {}
        Some(_) => {}
    }
    match *box_opt_void {
        None => {}
        _ => {}
    }
}
