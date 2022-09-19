#![feature(rustc_attrs)]

#[rustc_effective_visibility]
mod outer { //~ ERROR Public: pub(access_levels), Exported: pub(access_levels), Reachable: pub(access_levels), ReachableFromImplTrait: pub(access_levels)
    #[rustc_effective_visibility]
    pub mod inner1 { //~ ERROR Public: pub(access_levels), Exported: pub, Reachable: pub, ReachableFromImplTrait: pub

        #[rustc_effective_visibility]
        extern "C" {} //~ ERROR Public: pub(access_levels), Exported: pub, Reachable: pub, ReachableFromImplTrait: pub

        #[rustc_effective_visibility]
        pub trait PubTrait { //~ ERROR Public: pub(access_levels), Exported: pub, Reachable: pub, ReachableFromImplTrait: pub
            #[rustc_effective_visibility]
            const A: i32; //~ ERROR Public: pub(access_levels), Exported: pub, Reachable: pub, ReachableFromImplTrait: pub
            #[rustc_effective_visibility]
            type B; //~ ERROR Public: pub(access_levels), Exported: pub, Reachable: pub, ReachableFromImplTrait: pub
        }

        #[rustc_effective_visibility]
        struct PrivStruct; //~ ERROR Public: pub(inner1), Exported: pub(inner1), Reachable: pub(inner1), ReachableFromImplTrait: pub(inner1)

        #[rustc_effective_visibility]
        pub union PubUnion { //~ ERROR Public: pub(access_levels), Exported: pub, Reachable: pub, ReachableFromImplTrait: pub
            #[rustc_effective_visibility]
            a: u8, //~ ERROR Public: pub(inner1), Exported: pub(inner1), Reachable: pub(inner1), ReachableFromImplTrait: pub(inner1)
            #[rustc_effective_visibility]
            pub b: u8, //~ ERROR Public: pub(access_levels), Exported: pub, Reachable: pub, ReachableFromImplTrait: pub
        }

        #[rustc_effective_visibility]
        pub enum Enum { //~ ERROR Public: pub(access_levels), Exported: pub, Reachable: pub, ReachableFromImplTrait: pub
            #[rustc_effective_visibility]
            A( //~ ERROR Public: pub(access_levels), Exported: pub, Reachable: pub, ReachableFromImplTrait: pub
                #[rustc_effective_visibility]
                PubUnion,   //~ ERROR Public: pub(access_levels), Exported: pub, Reachable: pub, ReachableFromImplTrait: pub
            ),
        }
    }

    #[rustc_effective_visibility]
    macro_rules! none_macro { //~ Public: pub(access_levels), Exported: pub(access_levels), Reachable: pub(access_levels), ReachableFromImplTrait: pub(access_levels)
        () => {};
    }

    #[macro_export]
    #[rustc_effective_visibility]
    macro_rules! public_macro {  //~ Public: pub, Exported: pub, Reachable: pub, ReachableFromImplTrait: pub
        () => {};
    }

    #[rustc_effective_visibility]
    pub struct ReachableStruct { //~ ERROR Public: pub(access_levels), Exported: pub(access_levels), Reachable: pub, ReachableFromImplTrait: pub
        #[rustc_effective_visibility]
        pub a: u8, //~ ERROR Public: pub(access_levels), Exported: pub(access_levels), Reachable: pub, ReachableFromImplTrait: pub
    }
}

pub use outer::inner1;

pub fn foo() -> outer::ReachableStruct { outer::ReachableStruct {a: 0} }

fn main() {}
