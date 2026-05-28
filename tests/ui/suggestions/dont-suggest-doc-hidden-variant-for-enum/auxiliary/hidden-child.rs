#![crate_type = "lib"]

extern crate core;

pub mod __private {
    #[doc(hidden)]
    pub use core::option::Option::{self, None, Some};
}
