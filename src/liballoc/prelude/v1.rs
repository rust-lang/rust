//! The first version of the prelude of `alloc` crate.
//!
//! See the [module-level documentation](../index.html) for more.

#![unstable(feature = "alloc", issue = "27783")]

#[unstable(feature = "alloc", issue = "27783")] pub use crate::borrow::ToOwned;
#[unstable(feature = "alloc", issue = "27783")] pub use crate::boxed::Box;
#[unstable(feature = "alloc", issue = "27783")] pub use crate::slice::SliceConcatExt;
#[unstable(feature = "alloc", issue = "27783")] pub use crate::string::{String, ToString};
#[unstable(feature = "alloc", issue = "27783")] pub use crate::vec::Vec;
