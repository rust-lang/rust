//! The first version of the prelude of `alloc` crate.
//!
//! See the [module-level documentation](../index.html) for more.

#![unstable(feature = "alloc_prelude", issue = "58935")]

#[unstable(feature = "alloc_prelude", issue = "58935")] pub use crate::borrow::ToOwned;
#[unstable(feature = "alloc_prelude", issue = "58935")] pub use crate::boxed::Box;
#[unstable(feature = "alloc_prelude", issue = "58935")] pub use crate::slice::SliceConcatExt;
#[unstable(feature = "alloc_prelude", issue = "58935")] pub use crate::string::{String, ToString};
#[unstable(feature = "alloc_prelude", issue = "58935")] pub use crate::vec::Vec;
