// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Experimental extensions to `std` for Windows.
//!
//! For now, this module is limited to extracting handles, file
//! descriptors, and sockets, but its functionality will grow over
//! time.

#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "rust1", since = "1.0.0")]
pub mod io {
    use fs;
    use libc;
    use net;
    use sys_common::{net2, AsInner, FromInner};
    use sys;

    #[allow(deprecated)]
    use old_io;

    /// Raw HANDLEs.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub type RawHandle = libc::HANDLE;

    /// Raw SOCKETs.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub type RawSocket = libc::SOCKET;

    /// Extract raw handles.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub trait AsRawHandle {
        /// Extracts the raw handle, without taking any ownership.
        #[stable(feature = "rust1", since = "1.0.0")]
        fn as_raw_handle(&self) -> RawHandle;
    }

    /// Construct I/O objects from raw handles.
    #[unstable(feature = "from_raw_os",
               reason = "recent addition to the std::os::windows::io module")]
    pub trait FromRawHandle {
        /// Constructs a new I/O object from the specified raw handle.
        ///
        /// This function will **consume ownership** of the handle given,
        /// passing responsibility for closing the handle to the returned
        /// object.
        fn from_raw_handle(handle: RawHandle) -> Self;
    }

    #[allow(deprecated)]
    #[stable(feature = "rust1", since = "1.0.0")]
    impl AsRawHandle for old_io::fs::File {
        fn as_raw_handle(&self) -> RawHandle {
            self.as_inner().handle()
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl AsRawHandle for fs::File {
        fn as_raw_handle(&self) -> RawHandle {
            self.as_inner().handle().raw()
        }
    }

    #[unstable(feature = "from_raw_os", reason = "trait is unstable")]
    impl FromRawHandle for fs::File {
        fn from_raw_handle(handle: RawHandle) -> fs::File {
            fs::File::from_inner(sys::fs2::File::from_inner(handle))
        }
    }

    #[allow(deprecated)]
    #[stable(feature = "rust1", since = "1.0.0")]
    impl AsRawHandle for old_io::pipe::PipeStream {
        fn as_raw_handle(&self) -> RawHandle {
            self.as_inner().handle()
        }
    }

    #[allow(deprecated)]
    #[stable(feature = "rust1", since = "1.0.0")]
    impl AsRawHandle for old_io::net::pipe::UnixStream {
        fn as_raw_handle(&self) -> RawHandle {
            self.as_inner().handle()
        }
    }

    #[allow(deprecated)]
    #[stable(feature = "rust1", since = "1.0.0")]
    impl AsRawHandle for old_io::net::pipe::UnixListener {
        fn as_raw_handle(&self) -> RawHandle {
            self.as_inner().handle()
        }
    }

    #[allow(deprecated)]
    #[stable(feature = "rust1", since = "1.0.0")]
    impl AsRawHandle for old_io::net::pipe::UnixAcceptor {
        fn as_raw_handle(&self) -> RawHandle {
            self.as_inner().handle()
        }
    }

    /// Extract raw sockets.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub trait AsRawSocket {
        /// Extracts the underlying raw socket from this object.
        #[stable(feature = "rust1", since = "1.0.0")]
        fn as_raw_socket(&self) -> RawSocket;
    }

    /// Create I/O objects from raw sockets.
    #[unstable(feature = "from_raw_os", reason = "recent addition to module")]
    pub trait FromRawSocket {
        /// Creates a new I/O object from the given raw socket.
        ///
        /// This function will **consume ownership** of the socket provided and
        /// it will be closed when the returned object goes out of scope.
        fn from_raw_socket(sock: RawSocket) -> Self;
    }

    #[allow(deprecated)]
    #[stable(feature = "rust1", since = "1.0.0")]
    impl AsRawSocket for old_io::net::tcp::TcpStream {
        fn as_raw_socket(&self) -> RawSocket {
            self.as_inner().fd()
        }
    }

    #[allow(deprecated)]
    #[stable(feature = "rust1", since = "1.0.0")]
    impl AsRawSocket for old_io::net::tcp::TcpListener {
        fn as_raw_socket(&self) -> RawSocket {
            self.as_inner().socket()
        }
    }

    #[allow(deprecated)]
    #[stable(feature = "rust1", since = "1.0.0")]
    impl AsRawSocket for old_io::net::tcp::TcpAcceptor {
        fn as_raw_socket(&self) -> RawSocket {
            self.as_inner().socket()
        }
    }

    #[allow(deprecated)]
    #[stable(feature = "rust1", since = "1.0.0")]
    impl AsRawSocket for old_io::net::udp::UdpSocket {
        fn as_raw_socket(&self) -> RawSocket {
            self.as_inner().fd()
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl AsRawSocket for net::TcpStream {
        fn as_raw_socket(&self) -> RawSocket {
            *self.as_inner().socket().as_inner()
        }
    }
    #[stable(feature = "rust1", since = "1.0.0")]
    impl AsRawSocket for net::TcpListener {
        fn as_raw_socket(&self) -> RawSocket {
            *self.as_inner().socket().as_inner()
        }
    }
    #[stable(feature = "rust1", since = "1.0.0")]
    impl AsRawSocket for net::UdpSocket {
        fn as_raw_socket(&self) -> RawSocket {
            *self.as_inner().socket().as_inner()
        }
    }

    #[unstable(feature = "from_raw_os", reason = "trait is unstable")]
    impl FromRawSocket for net::TcpStream {
        fn from_raw_socket(sock: RawSocket) -> net::TcpStream {
            let sock = sys::net::Socket::from_inner(sock);
            net::TcpStream::from_inner(net2::TcpStream::from_inner(sock))
        }
    }
    #[unstable(feature = "from_raw_os", reason = "trait is unstable")]
    impl FromRawSocket for net::TcpListener {
        fn from_raw_socket(sock: RawSocket) -> net::TcpListener {
            let sock = sys::net::Socket::from_inner(sock);
            net::TcpListener::from_inner(net2::TcpListener::from_inner(sock))
        }
    }
    #[unstable(feature = "from_raw_os", reason = "trait is unstable")]
    impl FromRawSocket for net::UdpSocket {
        fn from_raw_socket(sock: RawSocket) -> net::UdpSocket {
            let sock = sys::net::Socket::from_inner(sock);
            net::UdpSocket::from_inner(net2::UdpSocket::from_inner(sock))
        }
    }
}

/// Windows-specific extensions to the primitives in the `std::ffi` module.
#[stable(feature = "rust1", since = "1.0.0")]
pub mod ffi {
    use ffi::{OsString, OsStr};
    use sys::os_str::Buf;
    use sys_common::wtf8::Wtf8Buf;
    use sys_common::{FromInner, AsInner};

    pub use sys_common::wtf8::EncodeWide;

    /// Windows-specific extensions to `OsString`.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub trait OsStringExt {
        /// Creates an `OsString` from a potentially ill-formed UTF-16 slice of
        /// 16-bit code units.
        ///
        /// This is lossless: calling `.encode_wide()` on the resulting string
        /// will always return the original code units.
        #[stable(feature = "rust1", since = "1.0.0")]
        fn from_wide(wide: &[u16]) -> Self;
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl OsStringExt for OsString {
        fn from_wide(wide: &[u16]) -> OsString {
            FromInner::from_inner(Buf { inner: Wtf8Buf::from_wide(wide) })
        }
    }

    /// Windows-specific extensions to `OsStr`.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub trait OsStrExt {
        /// Re-encodes an `OsStr` as a wide character sequence,
        /// i.e. potentially ill-formed UTF-16.
        ///
        /// This is lossless. Note that the encoding does not include a final
        /// null.
        #[stable(feature = "rust1", since = "1.0.0")]
        fn encode_wide(&self) -> EncodeWide;
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl OsStrExt for OsStr {
        fn encode_wide(&self) -> EncodeWide {
            self.as_inner().inner.encode_wide()
        }
    }
}

/// Windows-specific extensions for the primitives in `std::fs`
#[unstable(feature = "fs_ext", reason = "may require more thought/methods")]
pub mod fs {
    use fs::OpenOptions;
    use sys_common::AsInnerMut;

    /// Windows-specific extensions to `OpenOptions`
    pub trait OpenOptionsExt {
        /// Overrides the `dwDesiredAccess` argument to the call to `CreateFile`
        /// with the specified value.
        fn desired_access(&mut self, access: i32) -> &mut Self;

        /// Overrides the `dwCreationDisposition` argument to the call to
        /// `CreateFile` with the specified value.
        ///
        /// This will override any values of the standard `create` flags, for
        /// example.
        fn creation_disposition(&mut self, val: i32) -> &mut Self;

        /// Overrides the `dwFlagsAndAttributes` argument to the call to
        /// `CreateFile` with the specified value.
        ///
        /// This will override any values of the standard flags on the
        /// `OpenOptions` structure.
        fn flags_and_attributes(&mut self, val: i32) -> &mut Self;

        /// Overrides the `dwShareMode` argument to the call to `CreateFile` with
        /// the specified value.
        ///
        /// This will override any values of the standard flags on the
        /// `OpenOptions` structure.
        fn share_mode(&mut self, val: i32) -> &mut Self;
    }

    impl OpenOptionsExt for OpenOptions {
        fn desired_access(&mut self, access: i32) -> &mut OpenOptions {
            self.as_inner_mut().desired_access(access); self
        }
        fn creation_disposition(&mut self, access: i32) -> &mut OpenOptions {
            self.as_inner_mut().creation_disposition(access); self
        }
        fn flags_and_attributes(&mut self, access: i32) -> &mut OpenOptions {
            self.as_inner_mut().flags_and_attributes(access); self
        }
        fn share_mode(&mut self, access: i32) -> &mut OpenOptions {
            self.as_inner_mut().share_mode(access); self
        }
    }
}

/// A prelude for conveniently writing platform-specific code.
///
/// Includes all extension traits, and some important type definitions.
#[stable(feature = "rust1", since = "1.0.0")]
pub mod prelude {
    #[doc(no_inline)]
    pub use super::io::{RawSocket, RawHandle, AsRawSocket, AsRawHandle};
    #[doc(no_inline)] #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::ffi::{OsStrExt, OsStringExt};
    #[doc(no_inline)]
    pub use super::fs::OpenOptionsExt;
}
