//! The core prelude
//!
//! This module is intended for users of core which do not link to std as well.
//! This module is imported by default when `#![no_std]` is used in the same
//! manner as the standard library's prelude.

// No formatting: this file is nothing but re-exports, and their order is worth preserving.
#![cfg_attr(rustfmt, rustfmt::skip)]

#![stable(feature = "core_prelude", since = "1.4.0")]

mod common;

/// The first version of the prelude of The Rust Standard Library.
///
/// See the [module-level documentation](self) for more.
#[stable(feature = "rust1", since = "1.0.0")]
pub mod v1 {
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::common::*;

    // Do not `doc(inline)` these `doc(hidden)` items.
    #[unstable(
        feature = "rustc_encodable_decodable",
        issue = "none",
        soft,
        reason = "derive macro for `rustc-serialize`; should not be used in new code"
    )]
    #[allow(deprecated)]
    pub use crate::macros::builtin::{RustcDecodable, RustcEncodable};
}

/// The 2015 version of the core prelude.
///
/// See the [module-level documentation](self) for more.
#[stable(feature = "prelude_2015", since = "1.55.0")]
pub mod rust_2015 {
    #[stable(feature = "prelude_2015", since = "1.55.0")]
    #[doc(no_inline)]
    pub use super::v1::*;
}

/// The 2018 version of the core prelude.
///
/// See the [module-level documentation](self) for more.
#[stable(feature = "prelude_2018", since = "1.55.0")]
pub mod rust_2018 {
    #[stable(feature = "prelude_2018", since = "1.55.0")]
    #[doc(no_inline)]
    pub use super::v1::*;
}

/// The 2021 version of the core prelude.
///
/// See the [module-level documentation](self) for more.
#[stable(feature = "prelude_2021", since = "1.55.0")]
pub mod rust_2021 {
    #[stable(feature = "prelude_2021", since = "1.55.0")]
    #[doc(no_inline)]
    pub use super::v1::*;

    #[stable(feature = "prelude_2021", since = "1.55.0")]
    #[doc(no_inline)]
    pub use crate::iter::FromIterator;

    #[stable(feature = "prelude_2021", since = "1.55.0")]
    #[doc(no_inline)]
    pub use crate::convert::{TryFrom, TryInto};
}

/// The 2024 version of the core prelude.
///
/// See the [module-level documentation](self) for more.
#[stable(feature = "prelude_2024", since = "1.85.0")]
pub mod rust_2024 {
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::common::*;

    #[stable(feature = "prelude_2021", since = "1.55.0")]
    #[doc(no_inline)]
    pub use crate::iter::FromIterator;

    #[stable(feature = "prelude_2021", since = "1.55.0")]
    #[doc(no_inline)]
    pub use crate::convert::{TryFrom, TryInto};

    #[stable(feature = "prelude_2024", since = "1.85.0")]
    #[doc(no_inline)]
    pub use crate::future::{Future, IntoFuture};
}
