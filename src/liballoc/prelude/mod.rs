//! The alloc Prelude
//!
//! The purpose of this module is to alleviate imports of commonly-used
//! items of the `alloc` crate by adding a glob import to the top of modules:
//!
//! ```
//! # #![allow(unused_imports)]
//! # #![feature(alloc)]
//! extern crate alloc;
//! use alloc::prelude::v1::*;
//! ```

#![unstable(feature = "alloc", issue = "27783")]

pub mod v1;
