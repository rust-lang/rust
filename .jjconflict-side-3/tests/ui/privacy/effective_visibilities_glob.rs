// Effective visibility tracking for imports is fine-grained, so `S2` is not fully exported
// even if its parent import (`m::*`) is fully exported as a `use` item.

#![feature(rustc_attrs)]

mod m {
    #[rustc_effective_visibility]
    pub struct S1 {} //~ ERROR Direct: pub(crate), Reexported: pub, Reachable: pub, ReachableThroughImplTrait: pub
    #[rustc_effective_visibility]
    pub struct S2 {} //~ ERROR Direct: pub(crate), Reexported: pub(crate), Reachable: pub(crate), ReachableThroughImplTrait: pub(crate)
}

mod glob {
    #[rustc_effective_visibility]
    pub use crate::m::*; //~ ERROR Direct: pub(crate), Reexported: pub, Reachable: pub, ReachableThroughImplTrait: pub
}

#[rustc_effective_visibility]
pub use glob::S1; //~ ERROR Direct: pub, Reexported: pub, Reachable: pub, ReachableThroughImplTrait: pub

fn main() {}
