//! Test stability levels for items formerly dependent on multiple unstable features.
//@ aux-build:mixed-levels.rs

extern crate mixed_levels;

const USE_STABLE: () = mixed_levels::const_stable_fn();
const USE_UNSTABLE: () = mixed_levels::const_unstable_fn();
//~^ ERROR `const_unstable_fn` is not yet stable as a const fn

fn main() {
    mixed_levels::stable_mac!();
    mixed_levels::unstable_mac!(); //~ ERROR use of unstable library feature `unstable_a` [E0658]
}
