// Regression test for the 1.96 -> 1.97 stable-to-stable regression: an item exported
// only through a public glob, and also glob-imported with restricted visibility through
// a private facade, lost its exported effective visibility while name resolution still
// exported it. Downstream: spurious dead_code in this crate, "missing optimized MIR" in
// dependent crates (see ambiguous-import-visibility-globglob-mir.rs). The public glob
// declaration must drive the effective visibility of the whole reexport chain.

#![feature(rustc_attrs)]
#![allow(internal_features)]
#![deny(dead_code)]

mod inner {
    #[rustc_effective_visibility]
    pub fn f() {} //~ ERROR Direct: pub(crate), Reexported: pub, Reachable: pub, ReachableThroughImplTrait: pub
}

mod facade {
    #[allow(unused_imports)]
    pub(crate) use super::inner::f;
}

#[allow(unused_imports)]
use facade::*;
pub use inner::*;

fn main() {}
