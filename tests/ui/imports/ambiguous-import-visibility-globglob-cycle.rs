// issue: rust-lang/rust#160685
// Mutual globs of the same item cycle through `ambiguity_vis_max`.

#![feature(rustc_attrs)]
#![allow(internal_features)]
#![deny(dead_code)]

pub mod axiomatic {
    use super::*; // not pub
    pub use self::own::*;

    pub mod own {
        pub use super::*;
        pub use super::orphan::*;
    }

    pub mod orphan {
        pub use super::private::CollectionDescriptor;
    }

    mod private {
        #[rustc_effective_visibility]
        pub struct CollectionDescriptor {}
        //~^ ERROR Direct: pub(in crate::axiomatic), Reexported: pub, Reachable: pub, ReachableThroughImplTrait: pub
    }
}

pub use axiomatic::orphan::*;

fn main() {}
