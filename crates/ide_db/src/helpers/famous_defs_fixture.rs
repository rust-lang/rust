//- /libcore.rs crate:core
//! Signatures of traits, types and functions from the core lib for use in tests.
pub mod prelude {
    pub mod rust_2018 {
        pub use crate::{
            cmp::Ord,
            convert::{From, Into},
            default::Default,
            iter::{IntoIterator, Iterator},
            ops::{Fn, FnMut, FnOnce},
            option::Option::{self, *},
        };
    }
}
#[prelude_import]
pub use prelude::rust_2018::*;
//- /libstd.rs crate:std deps:core
//! Signatures of traits, types and functions from the std lib for use in tests.

/// Docs for return_keyword
mod return_keyword {}

/// Docs for prim_str
mod prim_str {}

pub use core::ops;
