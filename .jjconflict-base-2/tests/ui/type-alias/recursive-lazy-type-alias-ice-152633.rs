//! Ensure a self-referencing lazy type alias with `gca_min_const_items`
//! doesn't ICE during normalization.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/152633>.

#![feature(checked_type_aliases, gca_min_const_items, gca_macroless_args)]

trait Trait {
    #[rustc_always_gca]
    const ASSOC: ();
}
type Arr2 = [usize; <Arr2 as Trait>::ASSOC]; //~ ERROR E0275

fn main() {}
