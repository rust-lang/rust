//! Multi-segment associated constant paths in patterns (e.g. `Owner::K`) should
//! not get the misleading "interpreted as constant, not a new binding" label or
//! the "introduce a new binding instead" suggestion, since a multi-segment path
//! can never be a binding pattern.
//!
//! Single-segment paths that resolve to associated constants (e.g. via
//! `use Trait::K`) should still get both the label and the suggestion.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/153926>.

//@ dont-require-annotations: NOTE

#![feature(import_trait_associated_functions)]

struct Owner;

impl Owner {
    const K: (i32, i32) = (4, 2);
}

// Multi-segment assoc const path: no binding label or suggestion.
fn via_assoc_const(source: i32) {
    match source {
        Owner::K => {}
        //~^ ERROR mismatched types
        _ => {}
    }
}

trait Trait {
    const J: (i32, i32);
}

impl Trait for Owner {
    const J: (i32, i32) = (4, 2);
}

// Fully-qualified UFCS syntax: also multi-segment, no binding label or suggestion.
fn via_ufcs(source: i32) {
    match source {
        <Owner as Trait>::J => {}
        //~^ ERROR mismatched types
        _ => {}
    }
}

// Module-qualified free constant: multi-segment Resolved path, no binding label or suggestion.
mod inner {
    pub const C: (i32, i32) = (4, 2);
}

fn via_module_const(source: i32) {
    match source {
        inner::C => {}
        //~^ ERROR mismatched types
        _ => {}
    }
}

// Free constant with single-segment path: should keep the label and suggestion.
const FREE: (i32, i32) = (4, 2);

fn via_free_const(source: i32) {
    match source {
        FREE => {}
        //~^ ERROR mismatched types
        //~| NOTE `FREE` is interpreted as a constant, not a new binding
        _ => {}
    }
}

// Single-segment assoc const imported via `use Trait::J`: should keep the diagnostics.
use Trait::J;

fn via_imported_assoc_const(source: i32) {
    match source {
        J => {}
        //~^ ERROR mismatched types
        //~| NOTE `J` is interpreted as an associated constant, not a new binding
        _ => {}
    }
}

fn main() {}
