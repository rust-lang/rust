#![feature(rustc_attrs)]

#[rustc_effective_visibility]
mod outer { //~ ERROR Public: pub(crate), Exported: pub(crate), Reachable: pub(crate), ReachableFromImplTrait: pub(crate)
    #[rustc_effective_visibility]
    pub mod inner1 { //~ ERROR Public: pub(crate), Exported: pub, Reachable: pub, ReachableFromImplTrait: pub

        #[rustc_effective_visibility]
        extern "C" {} //~ ERROR Public: pub(crate), Exported: pub, Reachable: pub, ReachableFromImplTrait: pub

        #[rustc_effective_visibility]
        pub trait PubTrait { //~ ERROR Public: pub(crate), Exported: pub, Reachable: pub, ReachableFromImplTrait: pub
            #[rustc_effective_visibility]
            const A: i32; //~ ERROR Public: pub(crate), Exported: pub, Reachable: pub, ReachableFromImplTrait: pub
            #[rustc_effective_visibility]
            type B; //~ ERROR Public: pub(crate), Exported: pub, Reachable: pub, ReachableFromImplTrait: pub
        }

        #[rustc_effective_visibility]
        struct PrivStruct; //~ ERROR not in the table

        #[rustc_effective_visibility]
        pub union PubUnion { //~ ERROR Public: pub(crate), Exported: pub, Reachable: pub, ReachableFromImplTrait: pub
            #[rustc_effective_visibility]
            a: u8, //~ ERROR not in the table
            #[rustc_effective_visibility]
            pub b: u8, //~ ERROR Public: pub(crate), Exported: pub, Reachable: pub, ReachableFromImplTrait: pub
        }

        #[rustc_effective_visibility]
        pub enum Enum { //~ ERROR Public: pub(crate), Exported: pub, Reachable: pub, ReachableFromImplTrait: pub
            #[rustc_effective_visibility]
            A( //~ ERROR Public: pub(crate), Exported: pub, Reachable: pub, ReachableFromImplTrait: pub
                #[rustc_effective_visibility]
                PubUnion,  //~ ERROR Public: pub(crate), Exported: pub, Reachable: pub, ReachableFromImplTrait: pub
            ),
        }
    }

    #[rustc_effective_visibility]
    macro_rules! none_macro { //~ ERROR Public: pub(crate), Exported: pub(crate), Reachable: pub(crate), ReachableFromImplTrait: pub(crate)
        () => {};
    }

    #[macro_export]
    #[rustc_effective_visibility]
    macro_rules! public_macro { //~ ERROR Public: pub, Exported: pub, Reachable: pub, ReachableFromImplTrait: pub
        () => {};
    }

    #[rustc_effective_visibility]
    pub struct ReachableStruct { //~ ERROR Public: pub(crate), Exported: pub(crate), Reachable: pub, ReachableFromImplTrait: pub
        #[rustc_effective_visibility]
        pub a: u8, //~ ERROR Public: pub(crate), Exported: pub(crate), Reachable: pub, ReachableFromImplTrait: pub
    }
}

#[rustc_effective_visibility]
pub use outer::inner1; //~ ERROR Public: pub, Exported: pub, Reachable: pub, ReachableFromImplTrait: pub

pub fn foo() -> outer::ReachableStruct { outer::ReachableStruct {a: 0} }

mod half_public_import {
    #[rustc_effective_visibility]
    pub type HalfPublicImport = u8; //~ ERROR Public: pub(crate), Exported: pub, Reachable: pub, ReachableFromImplTrait: pub
    #[rustc_effective_visibility]
    #[allow(non_upper_case_globals)]
    pub(crate) const HalfPublicImport: u8 = 0; //~ ERROR Public: pub(crate), Exported: pub(crate), Reachable: pub(crate), ReachableFromImplTrait: pub(crate)
}

#[rustc_effective_visibility]
pub use half_public_import::HalfPublicImport; //~ ERROR Public: pub, Exported: pub, Reachable: pub, ReachableFromImplTrait: pub
                                              //~^ ERROR Public: pub, Exported: pub, Reachable: pub, ReachableFromImplTrait: pub

fn main() {}
