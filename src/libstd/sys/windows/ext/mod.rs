// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Platform-specific extensions to `std` for Windows.
//!
//! Provides access to platform-level information for Windows, and exposes
//! Windows-specific idioms that would otherwise be inappropriate as part
//! the core `std` library. These extensions allow developers to use
//! `std` types and idioms with Windows in a way that the normal
//! platform-agnostic idioms would not normally support.

#![stable(feature = "rust1", since = "1.0.0")]
#![doc(cfg(windows))]
#![allow(missing_docs)]

pub mod ffi;
pub mod fs;
pub mod io;
pub mod raw;
pub mod process;
pub mod thread;

/// A prelude for conveniently writing platform-specific code.
///
/// Includes all extension traits, and some important type definitions.
#[stable(feature = "rust1", since = "1.0.0")]
pub mod prelude {
    #[doc(no_inline)] #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::io::{RawSocket, RawHandle, AsRawSocket, AsRawHandle};
    #[doc(no_inline)] #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::io::{FromRawSocket, FromRawHandle, IntoRawSocket, IntoRawHandle};
    #[doc(no_inline)] #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::ffi::{OsStrExt, OsStringExt};
    #[doc(no_inline)] #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::fs::{OpenOptionsExt, MetadataExt};
    #[doc(no_inline)] #[stable(feature = "file_offset", since = "1.15.0")]
    pub use super::fs::FileExt;
}
