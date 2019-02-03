//! The alloc Prelude
//!
//! The purpose of this module is to alleviate imports of commonly-used
//! items of the `alloc` crate by adding a glob import to the top of modules:
//!
//! ```
//! # #![allow(unused_imports)]
//! # #![feature(alloc)]
//! extern crate alloc;
//! use alloc::prelude::*;
//! ```

#![unstable(feature = "alloc", issue = "27783")]

#[unstable(feature = "alloc", issue = "27783")] pub use crate::borrow::ToOwned;
#[unstable(feature = "alloc", issue = "27783")] pub use crate::boxed::Box;
#[unstable(feature = "alloc", issue = "27783")] pub use crate::slice::SliceConcatExt;
#[unstable(feature = "alloc", issue = "27783")] pub use crate::string::{String, ToString};
#[unstable(feature = "alloc", issue = "27783")] pub use crate::vec::Vec;
