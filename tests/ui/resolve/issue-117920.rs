#![crate_type = "lib"]

use super::A; //~ ERROR cannot find module

mod b {
    pub trait A {}
    pub trait B {}
}

/// [`A`]
pub use b::*;
