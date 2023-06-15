#![feature(rustc_attrs)]
#![allow(private_in_public)]

struct SemiPriv;

mod m {
    #[rustc_effective_visibility]
    struct Priv;
    //~^ ERROR not in the table
    //~| ERROR not in the table

    #[rustc_effective_visibility]
    pub fn foo() {} //~ ERROR Direct: pub(crate), Reexported: pub(crate), Reachable: pub(crate), ReachableThroughImplTrait: pub(crate)

    #[rustc_effective_visibility]
    impl crate::SemiPriv { //~ ERROR Direct: pub(crate), Reexported: pub(crate), Reachable: pub(crate), ReachableThroughImplTrait: pub(crate)
        pub fn f(_: Priv) {}
    }
}

fn main() {}
