// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! OS-specific functionality.

#![stable(feature = "os", since = "1.0.0")]
#![allow(missing_docs, bad_style, missing_debug_implementations)]

#[cfg(all(not(dox), any(target_os = "redox", unix)))]
#[stable(feature = "rust1", since = "1.0.0")]
pub use sys::ext as unix;
#[cfg(all(not(dox), windows))]
#[stable(feature = "rust1", since = "1.0.0")]
pub use sys::ext as windows;

#[cfg(dox)]
#[stable(feature = "rust1", since = "1.0.0")]
pub use sys::unix_ext as unix;
#[cfg(dox)]
#[stable(feature = "rust1", since = "1.0.0")]
pub use sys::windows_ext as windows;

#[cfg(any(dox, target_os = "linux"))]
#[doc(cfg(target_os = "linux"))]
pub mod linux;

#[cfg(all(not(dox), target_os = "android"))]    pub mod android;
#[cfg(all(not(dox), target_os = "bitrig"))]     pub mod bitrig;
#[cfg(all(not(dox), target_os = "dragonfly"))]  pub mod dragonfly;
#[cfg(all(not(dox), target_os = "freebsd"))]    pub mod freebsd;
#[cfg(all(not(dox), target_os = "haiku"))]      pub mod haiku;
#[cfg(all(not(dox), target_os = "ios"))]        pub mod ios;
#[cfg(all(not(dox), target_os = "macos"))]      pub mod macos;
#[cfg(all(not(dox), target_os = "nacl"))]       pub mod nacl;
#[cfg(all(not(dox), target_os = "netbsd"))]     pub mod netbsd;
#[cfg(all(not(dox), target_os = "openbsd"))]    pub mod openbsd;
#[cfg(all(not(dox), target_os = "solaris"))]    pub mod solaris;
#[cfg(all(not(dox), target_os = "emscripten"))] pub mod emscripten;
#[cfg(all(not(dox), target_os = "fuchsia"))]    pub mod fuchsia;

pub mod raw;
