// issue: rust-lang/rust#159038
// A restricted glob in the slot must not hide a more public glob of the same item.

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
