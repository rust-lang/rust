//! The libcore prelude
//!
//! This module is intended for users of libcore which do not link to libstd as
//! well. This module is imported by default when `#![no_std]` is used in the
//! same manner as the standard library's prelude.

#![stable(feature = "core_prelude", since = "1.4.0")]

pub mod v1;

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
