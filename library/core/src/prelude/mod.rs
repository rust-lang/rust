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
#[unstable(feature = "prelude_2015", issue = "85684")]
pub mod rust_2015 {
    #[unstable(feature = "prelude_2015", issue = "85684")]
    #[doc(no_inline)]
    pub use super::v1::*;
}

/// The 2018 version of the core prelude.
///
/// See the [module-level documentation](self) for more.
#[unstable(feature = "prelude_2018", issue = "85684")]
pub mod rust_2018 {
    #[unstable(feature = "prelude_2018", issue = "85684")]
    #[doc(no_inline)]
    pub use super::v1::*;
}

/// The 2021 version of the core prelude.
///
/// See the [module-level documentation](self) for more.
#[unstable(feature = "prelude_2021", issue = "85684")]
pub mod rust_2021 {
    #[unstable(feature = "prelude_2021", issue = "85684")]
    #[doc(no_inline)]
    pub use super::v1::*;

    #[unstable(feature = "prelude_2021", issue = "85684")]
    #[doc(no_inline)]
    pub use crate::iter::FromIterator;

    #[unstable(feature = "prelude_2021", issue = "85684")]
    #[doc(no_inline)]
    pub use crate::convert::{TryFrom, TryInto};
}
