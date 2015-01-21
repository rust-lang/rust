// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Experimental extensions to `std` for Unix platforms.
//!
//! For now, this module is limited to extracting file descriptors,
//! but its functionality will grow over time.
//!
//! # Example
//!
//! ```rust,ignore
//! #![feature(globs)]
//!
//! use std::io::fs::File;
//! use std::os::unix::prelude::*;
//!
//! fn main() {
//!     let f = File::create(&Path::new("foo.txt")).unwrap();
//!     let fd = f.as_raw_fd();
//!
//!     // use fd with native unix bindings
//! }
//! ```

#![unstable]

use vec::Vec;
use sys::os_str::Buf;
use sys_common::{AsInner, IntoInner, FromInner};
use ffi::{OsStr, OsString};
use libc;

use io;

/// Raw file descriptors.
pub type Fd = libc::c_int;

/// Extract raw file descriptor
pub trait AsRawFd {
    /// Extract the raw file descriptor, without taking any ownership.
    fn as_raw_fd(&self) -> Fd;
}

impl AsRawFd for io::fs::File {
    fn as_raw_fd(&self) -> Fd {
        self.as_inner().fd()
    }
}

impl AsRawFd for io::pipe::PipeStream {
    fn as_raw_fd(&self) -> Fd {
        self.as_inner().fd()
    }
}

impl AsRawFd for io::net::pipe::UnixStream {
    fn as_raw_fd(&self) -> Fd {
        self.as_inner().fd()
    }
}

impl AsRawFd for io::net::pipe::UnixListener {
    fn as_raw_fd(&self) -> Fd {
        self.as_inner().fd()
    }
}

impl AsRawFd for io::net::pipe::UnixAcceptor {
    fn as_raw_fd(&self) -> Fd {
        self.as_inner().fd()
    }
}

impl AsRawFd for io::net::tcp::TcpStream {
    fn as_raw_fd(&self) -> Fd {
        self.as_inner().fd()
    }
}

impl AsRawFd for io::net::tcp::TcpListener {
    fn as_raw_fd(&self) -> Fd {
        self.as_inner().fd()
    }
}

impl AsRawFd for io::net::tcp::TcpAcceptor {
    fn as_raw_fd(&self) -> Fd {
        self.as_inner().fd()
    }
}

impl AsRawFd for io::net::udp::UdpSocket {
    fn as_raw_fd(&self) -> Fd {
        self.as_inner().fd()
    }
}

// Unix-specific extensions to `OsString`.
pub trait OsStringExt {
    /// Create an `OsString` from a byte vector.
    fn from_vec(vec: Vec<u8>) -> Self;

    /// Yield the underlying byte vector of this `OsString`.
    fn into_vec(self) -> Vec<u8>;
}

impl OsStringExt for OsString {
    fn from_vec(vec: Vec<u8>) -> OsString {
        FromInner::from_inner(Buf { inner: vec })
    }

    fn into_vec(self) -> Vec<u8> {
        self.into_inner().inner
    }
}

// Unix-specific extensions to `OsStr`.
pub trait OsStrExt {
    fn as_byte_slice(&self) -> &[u8];
}

impl OsStrExt for OsStr {
    fn as_byte_slice(&self) -> &[u8] {
        &self.as_inner().inner
    }
}

/// A prelude for conveniently writing platform-specific code.
///
/// Includes all extension traits, and some important type definitions.
pub mod prelude {
    pub use super::{Fd, AsRawFd};
}
