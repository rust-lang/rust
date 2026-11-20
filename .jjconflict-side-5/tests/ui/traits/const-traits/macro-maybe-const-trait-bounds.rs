// Ensure that we don't consider `[` to begin trait bounds to contain breakages.
// Only `[const]` in its entirety begins a trait bound.
// See also test `macro-const-trait-bound-theoretical-regression.rs`.

//@ check-pass (KEEP THIS AS A PASSING TEST!)
// Setting the edition to >2015 since we didn't regress `check! { dyn [const] Trait }` in Rust 2015.
// See also test `traits/const-traits/macro-dyn-const-2015.rs`.
//@ edition:2018

macro_rules! check {
    ($ty:ty) => { compile_error!("ty"); }; // KEEP THIS RULE FIRST AND AS IS!

    // DON'T MODIFY THE MATCHERS BELOW UNLESS THE CONST TRAIT MODIFIER SYNTAX CHANGES!

    (dyn [$($any:tt)*] Trait) => { /* KEEP THIS EMPTY! */ };
    (impl [$($any:tt)*] Trait) => { /* KEEP THIS EMPTY! */ };
}

check!(dyn [T] Trait);

// issue: <https://github.com/rust-lang/rust/issues/146417>
check!(impl [T] Trait);
check!(impl [T: Bound] Trait);

fn main() {}
