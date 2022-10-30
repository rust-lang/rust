//! This module reexports various crates and modules from unstable rustc APIs.
//! Add anything you need here and it will get slowly transferred to a stable API.
//! Only use rustc_smir in your dependencies and use the reexports here instead of
//! directly referring to the unstable crates.

macro_rules! crates {
    ($($rustc_name:ident -> $name:ident,)*) => {
        $(
            #[cfg(not(feature = "default"))]
            pub extern crate $rustc_name as $name;
            #[cfg(feature = "default")]
            pub use $rustc_name as $name;
        )*
    }
}

crates! {
    rustc_borrowck -> borrowck,
    rustc_driver -> driver,
    rustc_hir -> hir,
    rustc_interface -> interface,
    rustc_middle -> middle,
    rustc_mir_dataflow -> dataflow,
    rustc_mir_transform -> transform,
    rustc_serialize -> serialize,
    rustc_trait_selection -> trait_selection,
}
