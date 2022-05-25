//! This module reexports various crates and modules from unstable rustc APIs.
//! Add anything you need here and it will get slowly transferred to a stable API.
//! Only use rustc_smir in your dependencies and use the reexports here instead of
//! directly referring to the unstable crates.

pub use rustc_borrowck as borrowck;
pub use rustc_driver as driver;
pub use rustc_hir as hir;
pub use rustc_interface as interface;
pub use rustc_middle as middle;
pub use rustc_mir_dataflow as dataflow;
pub use rustc_mir_transform as transform;
pub use rustc_serialize as serialize;
pub use rustc_trait_selection as trait_selection;
