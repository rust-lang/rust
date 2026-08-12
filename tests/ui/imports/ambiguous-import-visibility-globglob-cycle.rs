// Regression test for #160685: ambiguous glob imports whose reexport chains
// form a cycle (`axiomatic` and `own` glob-import each other, and the same
// item also arrives through `orphan`) sent `update_decl_chain` into infinite
// recursion through `ambiguity_vis_max`, overflowing the stack. Each
// declaration is now visited at most once per chain walk; the effective
// visibility fixpoint loop picks up anything a skipped revisit would have
// contributed. Minimized from the `reflect_tools` crate by @theemathas.

//@ check-pass
//@ edition: 2024

pub mod axiomatic {
    use super::*; // not pub
    pub use own::*;

    pub mod own {
        pub use super::*;
        pub use orphan::*;
    }

    pub mod orphan {
        pub use super::private::CollectionDescriptor;
    }

    pub mod private {
        pub struct CollectionDescriptor;
    }
}

pub use axiomatic::orphan::*;

fn main() {}
