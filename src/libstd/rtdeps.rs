// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module contains the linkage attributes to all runtime dependencies of
//! the standard library This varies per-platform, but these libraries are
//! necessary for running libstd.

#![cfg(not(cargobuild))]

// LLVM implements the `frem` instruction as a call to `fmod`, which lives in
// libm. Hence, we must explicitly link to it.
//
// On Linux, librt and libdl are indirect dependencies via std,
// and binutils 2.22+ won't add them automatically
#[cfg(all(target_os = "linux", not(target_env = "musl")))]
#[link(name = "dl")]
#[link(name = "pthread")]
extern {}

#[cfg(target_os = "android")]
#[link(name = "dl")]
#[link(name = "log")]
extern {}

#[cfg(target_os = "freebsd")]
#[link(name = "execinfo")]
#[link(name = "pthread")]
extern {}

#[cfg(any(target_os = "dragonfly",
          target_os = "bitrig",
          target_os = "netbsd",
          target_os = "openbsd"))]
#[link(name = "pthread")]
extern {}

#[cfg(target_os = "solaris")]
#[link(name = "socket")]
#[link(name = "posix4")]
#[link(name = "pthread")]
extern {}

// For PNaCl targets, nacl_io is a Pepper wrapper for some IO functions
// missing (ie always error) in Newlib.
#[cfg(all(target_os = "nacl", not(test)))]
#[link(name = "nacl_io", kind = "static")]
#[link(name = "c++", kind = "static")] // for `nacl_io` and EH.
#[link(name = "pthread", kind = "static")]
extern {}

#[cfg(target_os = "macos")]
#[link(name = "System")]
extern {}

#[cfg(target_os = "ios")]
#[link(name = "System")]
extern {}

#[cfg(target_os = "haiku")]
#[link(name = "network")]
extern {}
