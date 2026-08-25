#![crate_type = "lib"]

extern crate core;

#[doc(hidden)]
pub mod __private {
    pub use core::option::Option::{self, None, Some};
}
