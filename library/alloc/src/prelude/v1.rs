//! The first version of the prelude of `alloc` crate.
//!
//! See the [module-level documentation](../index.html) for more.

#![unstable(feature = "alloc_prelude", issue = "58935")]

#[allow(missing_docs)]
#[unstable(feature = "alloc_prelude", issue = "58935")]
pub use crate::borrow::ToOwned;
#[allow(missing_docs)]
#[unstable(feature = "alloc_prelude", issue = "58935")]
pub use crate::boxed::Box;
#[allow(missing_docs)]
#[unstable(feature = "alloc_prelude", issue = "58935")]
pub use crate::string::{String, ToString};
#[allow(missing_docs)]
#[unstable(feature = "alloc_prelude", issue = "58935")]
pub use crate::vec::Vec;
