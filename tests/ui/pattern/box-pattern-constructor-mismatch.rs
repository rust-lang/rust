//! Test that `box _` patterns and `Box { .. }` patterns can't be used to match on the same place.
//! This is required for the current implementation of exhaustiveness analysis for deref patterns.

#![feature(box_patterns)]

fn main() {
    match Box::new(0) {
        box _ => {} //~ ERROR mix of deref patterns and normal constructors
        Box { .. } => {}
    }
}
