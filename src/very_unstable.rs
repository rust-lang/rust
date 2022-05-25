//! This module reexports various crates and modules from unstable rustc APIs.
//! Add anything you need here and it will get slowly transferred to a stable API.
//! Only use rustc_smir in your dependencies and use the reexports here instead of
//! directly referring to the unstable crates.

pub use rustc_borrowck as borrowck;
pub use rustc_driver as driver;
pub use rustc_interface as interface;
pub use rustc_middle as middle;
