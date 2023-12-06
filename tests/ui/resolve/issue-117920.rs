#![crate_type = "lib"]

use super::A; //~ ERROR failed to resolve

mod b {
    pub trait A {}
    pub trait B {}
}

/// [`A`]
pub use b::*;
