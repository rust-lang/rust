// Ensure that the introduction of `const` and `[const]` trait bounds didn't regress this
// Rust 2015 code. See also test `macro-const-trait-bound-theoretical-regression.rs`.

//@ edition: 2015
//@ check-pass (KEEP THIS AS A PASSING TEST!)

macro_rules! check {
    ($ty:ty) => { compile_error!("ty"); }; // KEEP THIS RULE FIRST AND AS IS!

    // DON'T MODIFY THE MATCHERS BELOW UNLESS THE CONST TRAIT MODIFIER SYNTAX CHANGES!

    (dyn $c:ident) => { /* KEEP THIS EMPTY! */ };
    (dyn [$c:ident]) => { /* KEEP THIS EMPTY! */ };
}

check! { dyn const }
check! { dyn [const] }

fn main() {}
