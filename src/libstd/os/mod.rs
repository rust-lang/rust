// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! OS-specific functionality

#![stable(feature = "os", since = "1.0.0")]
#![allow(missing_docs, bad_style)]

#[cfg(unix)]
#[stable(feature = "rust1", since = "1.0.0")]
pub use sys::ext as unix;
#[cfg(windows)]
#[stable(feature = "rust1", since = "1.0.0")]
pub use sys::ext as windows;

#[cfg(target_os = "android")]   pub mod android;
#[cfg(target_os = "bitrig")]    pub mod bitrig;
#[cfg(target_os = "dragonfly")] pub mod dragonfly;
#[cfg(target_os = "freebsd")]   pub mod freebsd;
#[cfg(target_os = "ios")]       pub mod ios;
#[cfg(target_os = "linux")]     pub mod linux;
#[cfg(target_os = "macos")]     pub mod macos;
#[cfg(target_os = "nacl")]      pub mod nacl;
#[cfg(target_os = "netbsd")]   pub mod netbsd;
#[cfg(target_os = "openbsd")]   pub mod openbsd;

pub mod raw;
