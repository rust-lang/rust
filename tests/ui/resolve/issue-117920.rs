#![crate_type = "lib"]

use super::A; //~ ERROR cannot find

mod b {
    pub trait A {}
    pub trait B {}
}

/// [`A`]
pub use b::*;
