#![crate_type = "lib"]

use super::A; //~ ERROR too many leading `super` keywords within `crate`

mod b {
    pub trait A {}
    pub trait B {}
}

/// [`A`]
pub use b::*;
