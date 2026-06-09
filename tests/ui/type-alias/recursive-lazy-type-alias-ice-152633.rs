//! Ensure a self-referencing lazy type alias with `min_generic_const_args`
//! doesn't ICE during normalization.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/152633>.

#![feature(lazy_type_alias)]
#![feature(min_generic_const_args)]

trait Trait {
    type const ASSOC: ();
}
type Arr2 = [usize; <Arr2 as Trait>::ASSOC]; //~ ERROR E0275

fn main() {}
