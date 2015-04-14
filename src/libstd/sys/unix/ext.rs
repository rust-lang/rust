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
//! ```no_run
//! use std::fs::File;
//! use std::os::unix::prelude::*;
//!
//! fn main() {
//!     let f = File::create("foo.txt").unwrap();
//!     let fd = f.as_raw_fd();
//!
//!     // use fd with native unix bindings
//! }
//! ```

#![stable(feature = "rust1", since = "1.0.0")]

/// Unix-specific extensions to general I/O primitives
#[stable(feature = "rust1", since = "1.0.0")]
pub mod io {
    use fs;
    use libc;
    use net;
    use sys_common::{net2, AsInner, FromInner};
    use sys;

    /// Raw file descriptors.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub type RawFd = libc::c_int;

    /// A trait to extract the raw unix file descriptor from an underlying
    /// object.
    ///
    /// This is only available on unix platforms and must be imported in order
    /// to call the method. Windows platforms have a corresponding `AsRawHandle`
    /// and `AsRawSocket` set of traits.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub trait AsRawFd {
        /// Extract the raw file descriptor.
        ///
        /// This method does **not** pass ownership of the raw file descriptor
        /// to the caller. The descriptor is only guarantee to be valid while
        /// the original object has not yet been destroyed.
        #[stable(feature = "rust1", since = "1.0.0")]
        fn as_raw_fd(&self) -> RawFd;
    }

    /// A trait to express the ability to construct an object from a raw file
    /// descriptor.
    #[unstable(feature = "from_raw_os",
               reason = "recent addition to std::os::unix::io")]
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
        unsafe fn from_raw_fd(fd: RawFd) -> Self;
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl AsRawFd for fs::File {
        fn as_raw_fd(&self) -> RawFd {
            self.as_inner().fd().raw()
        }
    }
    #[unstable(feature = "from_raw_os", reason = "trait is unstable")]
    impl FromRawFd for fs::File {
        unsafe fn from_raw_fd(fd: RawFd) -> fs::File {
            fs::File::from_inner(sys::fs2::File::from_inner(fd))
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl AsRawFd for net::TcpStream {
        fn as_raw_fd(&self) -> RawFd { *self.as_inner().socket().as_inner() }
    }
    #[stable(feature = "rust1", since = "1.0.0")]
    impl AsRawFd for net::TcpListener {
        fn as_raw_fd(&self) -> RawFd { *self.as_inner().socket().as_inner() }
    }
    #[stable(feature = "rust1", since = "1.0.0")]
    impl AsRawFd for net::UdpSocket {
        fn as_raw_fd(&self) -> RawFd { *self.as_inner().socket().as_inner() }
    }

    #[unstable(feature = "from_raw_os", reason = "trait is unstable")]
    impl FromRawFd for net::TcpStream {
        unsafe fn from_raw_fd(fd: RawFd) -> net::TcpStream {
            let socket = sys::net::Socket::from_inner(fd);
            net::TcpStream::from_inner(net2::TcpStream::from_inner(socket))
        }
    }
    #[unstable(feature = "from_raw_os", reason = "trait is unstable")]
    impl FromRawFd for net::TcpListener {
        unsafe fn from_raw_fd(fd: RawFd) -> net::TcpListener {
            let socket = sys::net::Socket::from_inner(fd);
            net::TcpListener::from_inner(net2::TcpListener::from_inner(socket))
        }
    }
    #[unstable(feature = "from_raw_os", reason = "trait is unstable")]
    impl FromRawFd for net::UdpSocket {
        unsafe fn from_raw_fd(fd: RawFd) -> net::UdpSocket {
            let socket = sys::net::Socket::from_inner(fd);
            net::UdpSocket::from_inner(net2::UdpSocket::from_inner(socket))
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// OsString and OsStr
////////////////////////////////////////////////////////////////////////////////

/// Unix-specific extension to the primitives in the `std::ffi` module
#[stable(feature = "rust1", since = "1.0.0")]
pub mod ffi {
    use ffi::{OsStr, OsString};
    use mem;
    use prelude::v1::*;
    use sys::os_str::Buf;
    use sys_common::{FromInner, IntoInner, AsInner};

    /// Unix-specific extensions to `OsString`.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub trait OsStringExt {
        /// Create an `OsString` from a byte vector.
        #[stable(feature = "rust1", since = "1.0.0")]
        fn from_vec(vec: Vec<u8>) -> Self;

        /// Yield the underlying byte vector of this `OsString`.
        #[stable(feature = "rust1", since = "1.0.0")]
        fn into_vec(self) -> Vec<u8>;
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl OsStringExt for OsString {
        fn from_vec(vec: Vec<u8>) -> OsString {
            FromInner::from_inner(Buf { inner: vec })
        }
        fn into_vec(self) -> Vec<u8> {
            self.into_inner().inner
        }
    }

    /// Unix-specific extensions to `OsStr`.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub trait OsStrExt {
        #[stable(feature = "rust1", since = "1.0.0")]
        fn from_bytes(slice: &[u8]) -> &Self;

        /// Get the underlying byte view of the `OsStr` slice.
        #[stable(feature = "rust1", since = "1.0.0")]
        fn as_bytes(&self) -> &[u8];
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl OsStrExt for OsStr {
        fn from_bytes(slice: &[u8]) -> &OsStr {
            unsafe { mem::transmute(slice) }
        }
        fn as_bytes(&self) -> &[u8] {
            &self.as_inner().inner
        }
    }
}

/// Unix-specific extensions to primitives in the `std::fs` module.
#[unstable(feature = "fs_ext",
           reason = "may want a more useful mode abstraction")]
pub mod fs {
    use sys_common::{FromInner, AsInner, AsInnerMut};
    use fs::{Permissions, OpenOptions};

    /// Unix-specific extensions to `Permissions`
    pub trait PermissionsExt {
        fn mode(&self) -> i32;
        fn set_mode(&mut self, mode: i32);
    }

    impl PermissionsExt for Permissions {
        fn mode(&self) -> i32 { self.as_inner().mode() }

        fn set_mode(&mut self, mode: i32) {
            *self = FromInner::from_inner(FromInner::from_inner(mode));
        }
    }

    /// Unix-specific extensions to `OpenOptions`
    pub trait OpenOptionsExt {
        /// Set the mode bits that a new file will be created with.
        ///
        /// If a new file is created as part of a `File::open_opts` call then this
        /// specified `mode` will be used as the permission bits for the new file.
        fn mode(&mut self, mode: i32) -> &mut Self;
    }

    impl OpenOptionsExt for OpenOptions {
        fn mode(&mut self, mode: i32) -> &mut OpenOptions {
            self.as_inner_mut().mode(mode); self
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Process and Command
////////////////////////////////////////////////////////////////////////////////

/// Unix-specific extensions to primitives in the `std::process` module.
#[stable(feature = "rust1", since = "1.0.0")]
pub mod process {
    use prelude::v1::*;
    use libc::{uid_t, gid_t};
    use process;
    use sys;
    use sys_common::{AsInnerMut, AsInner};

    /// Unix-specific extensions to the `std::process::Command` builder
    #[stable(feature = "rust1", since = "1.0.0")]
    pub trait CommandExt {
        /// Sets the child process's user id. This translates to a
        /// `setuid` call in the child process. Failure in the `setuid`
        /// call will cause the spawn to fail.
        #[stable(feature = "rust1", since = "1.0.0")]
        fn uid(&mut self, id: uid_t) -> &mut process::Command;

        /// Similar to `uid`, but sets the group id of the child process. This has
        /// the same semantics as the `uid` field.
        #[stable(feature = "rust1", since = "1.0.0")]
        fn gid(&mut self, id: gid_t) -> &mut process::Command;
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl CommandExt for process::Command {
        fn uid(&mut self, id: uid_t) -> &mut process::Command {
            self.as_inner_mut().uid = Some(id);
            self
        }

        fn gid(&mut self, id: gid_t) -> &mut process::Command {
            self.as_inner_mut().gid = Some(id);
            self
        }
    }

    /// Unix-specific extensions to `std::process::ExitStatus`
    #[stable(feature = "rust1", since = "1.0.0")]
    pub trait ExitStatusExt {
        /// If the process was terminated by a signal, returns that signal.
        #[stable(feature = "rust1", since = "1.0.0")]
        fn signal(&self) -> Option<i32>;
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl ExitStatusExt for process::ExitStatus {
        fn signal(&self) -> Option<i32> {
            match *self.as_inner() {
                sys::process2::ExitStatus::Signal(s) => Some(s),
                _ => None
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Prelude
////////////////////////////////////////////////////////////////////////////////

/// A prelude for conveniently writing platform-specific code.
///
/// Includes all extension traits, and some important type definitions.
#[stable(feature = "rust1", since = "1.0.0")]
pub mod prelude {
    #[doc(no_inline)]
    pub use super::io::{RawFd, AsRawFd};
    #[doc(no_inline)] #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::ffi::{OsStrExt, OsStringExt};
    #[doc(no_inline)]
    pub use super::fs::{PermissionsExt, OpenOptionsExt};
    #[doc(no_inline)] #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::process::{CommandExt, ExitStatusExt};
}
