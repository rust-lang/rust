#![rustc_effective_visibility] //~ ERROR Direct: pub, Reexported: pub, Reachable: pub, ReachableThroughImplTrait: pub
#![feature(rustc_attrs)]

#[rustc_effective_visibility]
mod outer { //~ ERROR Direct: pub(crate), Reexported: pub(crate), Reachable: pub(crate), ReachableThroughImplTrait: pub(crate)
    #[rustc_effective_visibility]
    pub mod inner1 { //~ ERROR Direct: pub(crate), Reexported: pub, Reachable: pub, ReachableThroughImplTrait: pub

        #[rustc_effective_visibility]
        extern "C" {} //~ ERROR not in the table

        #[rustc_effective_visibility]
        pub trait PubTrait { //~ ERROR Direct: pub(crate), Reexported: pub, Reachable: pub, ReachableThroughImplTrait: pub
            #[rustc_effective_visibility]
            const A: i32; //~ ERROR Direct: pub(crate), Reexported: pub, Reachable: pub, ReachableThroughImplTrait: pub
            #[rustc_effective_visibility]
            type B; //~ ERROR Direct: pub(crate), Reexported: pub, Reachable: pub, ReachableThroughImplTrait: pub
        }

        #[rustc_effective_visibility]
        struct PrivStruct; //~ ERROR not in the table
                           //~| ERROR not in the table

        #[rustc_effective_visibility]
        pub union PubUnion { //~ ERROR Direct: pub(crate), Reexported: pub, Reachable: pub, ReachableThroughImplTrait: pub
            #[rustc_effective_visibility]
            a: u8, //~ ERROR not in the table
            #[rustc_effective_visibility]
            pub b: u8, //~ ERROR Direct: pub(crate), Reexported: pub, Reachable: pub, ReachableThroughImplTrait: pub
        }

        #[rustc_effective_visibility]
        pub enum Enum { //~ ERROR Direct: pub(crate), Reexported: pub, Reachable: pub, ReachableThroughImplTrait: pub
            #[rustc_effective_visibility]
            A( //~ ERROR Direct: pub(crate), Reexported: pub, Reachable: pub, ReachableThroughImplTrait: pub
               //~| ERROR Direct: pub(crate), Reexported: pub, Reachable: pub, ReachableThroughImplTrait: pub
                #[rustc_effective_visibility]
                PubUnion,  //~ ERROR Direct: pub(crate), Reexported: pub, Reachable: pub, ReachableThroughImplTrait: pub
            ),
        }
    }

    #[rustc_effective_visibility]
    macro_rules! none_macro { //~ ERROR not in the table
        () => {};
    }

    #[macro_export]
    #[rustc_effective_visibility]
    macro_rules! public_macro { //~ ERROR Direct: pub(self), Reexported: pub, Reachable: pub, ReachableThroughImplTrait: pub
        () => {};
    }

    #[rustc_effective_visibility]
    pub struct ReachableStruct { //~ ERROR Direct: pub(crate), Reexported: pub(crate), Reachable: pub, ReachableThroughImplTrait: pub
        #[rustc_effective_visibility]
        pub a: u8, //~ ERROR Direct: pub(crate), Reexported: pub(crate), Reachable: pub, ReachableThroughImplTrait: pub
    }
}

#[rustc_effective_visibility]
pub use outer::inner1; //~ ERROR Direct: pub, Reexported: pub, Reachable: pub, ReachableThroughImplTrait: pub

pub fn foo() -> outer::ReachableStruct { outer::ReachableStruct {a: 0} }

mod half_public_import {
    #[rustc_effective_visibility]
    pub type HalfPublicImport = u8; //~ ERROR Direct: pub(crate), Reexported: pub, Reachable: pub, ReachableThroughImplTrait: pub
    #[rustc_effective_visibility]
    #[allow(non_upper_case_globals)]
    pub(crate) const HalfPublicImport: u8 = 0; //~ ERROR Direct: pub(crate), Reexported: pub(crate), Reachable: pub(crate), ReachableThroughImplTrait: pub(crate)
}

#[rustc_effective_visibility]
pub use half_public_import::HalfPublicImport; //~ ERROR Direct: pub, Reexported: pub, Reachable: pub, ReachableThroughImplTrait: pub

fn main() {}
