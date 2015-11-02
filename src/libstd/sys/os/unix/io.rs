// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unix-specific extensions to general I/O primitives

#![stable(feature = "rust1", since = "1.0.0")]

use sys::inner::*;
use sys::fs as sys_fs;
use fs;
use net;
use os::raw;

/// Raw file descriptors.
#[stable(feature = "rust1", since = "1.0.0")]
pub type RawFd = raw::c_int;

/// A trait to extract the raw unix file descriptor from an underlying
/// object.
///
/// This is only available on unix platforms and must be imported in order
/// to call the method. Windows platforms have a corresponding `AsRawHandle`
/// and `AsRawSocket` set of traits.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait AsRawFd {
    /// Extracts the raw file descriptor.
    ///
    /// This method does **not** pass ownership of the raw file descriptor
    /// to the caller. The descriptor is only guarantee to be valid while
    /// the original object has not yet been destroyed.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn as_raw_fd(&self) -> RawFd;
}

/// A trait to express the ability to construct an object from a raw file
/// descriptor.
#[stable(feature = "from_raw_os", since = "1.1.0")]
pub trait FromRawFd {
    /// Constructs a new instances of `Self` from the given raw file
    /// descriptor.
    ///
    /// This function **consumes ownership** of the specified file
    /// descriptor. The returned object will take responsibility for closing
    /// it when the object goes out of scope.
    ///
    /// This function is also unsafe as the primitives currently returned
    /// have the contract that they are the sole owner of the file
    /// descriptor they are wrapping. Usage of this function could
    /// accidentally allow violating this contract which can cause memory
    /// unsafety in code that relies on it being true.
    #[stable(feature = "from_raw_os", since = "1.1.0")]
    unsafe fn from_raw_fd(fd: RawFd) -> Self;
}

/// A trait to express the ability to consume an object and acquire ownership of
/// its raw file descriptor.
#[stable(feature = "into_raw_os", since = "1.4.0")]
pub trait IntoRawFd {
    /// Consumes this object, returning the raw underlying file descriptor.
    ///
    /// This function **transfers ownership** of the underlying file descriptor
    /// to the caller. Callers are then the unique owners of the file descriptor
    /// and must close the descriptor once it's no longer needed.
    #[stable(feature = "into_raw_os", since = "1.4.0")]
    fn into_raw_fd(self) -> RawFd;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRawFd for fs::File {
    fn as_raw_fd(&self) -> RawFd {
        *self.as_inner().as_inner()
    }
}
#[stable(feature = "from_raw_os", since = "1.1.0")]
impl FromRawFd for fs::File {
    unsafe fn from_raw_fd(fd: RawFd) -> fs::File {
        fs::File::from_inner(sys_fs::File::from_inner(fd))
    }
}
#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for fs::File {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_inner()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRawFd for net::TcpStream {
    fn as_raw_fd(&self) -> RawFd { *self.as_inner().as_inner() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl AsRawFd for net::TcpListener {
    fn as_raw_fd(&self) -> RawFd { *self.as_inner().as_inner() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl AsRawFd for net::UdpSocket {
    fn as_raw_fd(&self) -> RawFd { *self.as_inner().as_inner() }
}

#[stable(feature = "from_raw_os", since = "1.1.0")]
impl FromRawFd for net::TcpStream {
    unsafe fn from_raw_fd(fd: RawFd) -> net::TcpStream {
        net::TcpStream::from_inner(FromInner::from_inner(fd))
    }
}
#[stable(feature = "from_raw_os", since = "1.1.0")]
impl FromRawFd for net::TcpListener {
    unsafe fn from_raw_fd(fd: RawFd) -> net::TcpListener {
        net::TcpListener::from_inner(FromInner::from_inner(fd))
    }
}
#[stable(feature = "from_raw_os", since = "1.1.0")]
impl FromRawFd for net::UdpSocket {
    unsafe fn from_raw_fd(fd: RawFd) -> net::UdpSocket {
        net::UdpSocket::from_inner(FromInner::from_inner(fd))
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for net::TcpStream {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_inner()
    }
}
#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for net::TcpListener {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_inner()
    }
}
#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for net::UdpSocket {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_inner()
    }
}
