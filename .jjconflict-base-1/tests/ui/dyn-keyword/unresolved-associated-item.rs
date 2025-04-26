//! Checks the E0782 label when a `Trait::item` path matches nothing on the trait.
//! There is no typo to suggest, so the label is the only thing pointing at the real mistake.
//!
//! This is the negative counterpart of `misspelled-associated-item.rs`, which covers the
//! near-miss arm.
//! Keep `nonexistent` far from `exists` so `find_best_match_for_name` returns `None`.
//! If the names get closer, this test silently turns into a copy of that one and still passes.
//!
//! The label is a `span_label`, not a sub-diagnostic, so it has no `// ~` annotation.
//! Only the `.stderr` checks it.
//!
//! See <https://github.com/rust-lang/rust/issues/136994>.

//@ edition: 2021
// E0782 only fires on 2021+. On 2018 this is just the bare-trait-object lint and the label
// never appears.

trait Trait {
    fn exists() -> Self;
}

fn main() {
    Trait::nonexistent();
    //~^ ERROR expected a type, found a trait
    //~| HELP you can add the `dyn` keyword if you want a trait object
}
