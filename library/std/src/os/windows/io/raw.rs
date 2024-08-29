//! Windows-specific extensions to general I/O primitives.

#![stable(feature = "rust1", since = "1.0.0")]

#[cfg(doc)]
use crate::os::windows::io::{AsHandle, AsSocket};
use crate::os::windows::io::{OwnedHandle, OwnedSocket};
use crate::os::windows::raw;
use crate::sys_common::{self, AsInner, FromInner, IntoInner};
use crate::{fs, io, net, ptr, sys};

/// Raw HANDLEs.
#[stable(feature = "rust1", since = "1.0.0")]
pub type RawHandle = raw::HANDLE;

/// Raw SOCKETs.
#[stable(feature = "rust1", since = "1.0.0")]
pub type RawSocket = raw::SOCKET;

/// Extracts raw handles.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait AsRawHandle {
    /// Extracts the raw handle.
    ///
    /// This function is typically used to **borrow** an owned handle.
    /// When used in this way, this method does **not** pass ownership of the
    /// raw handle to the caller, and the handle is only guaranteed
    /// to be valid while the original object has not yet been destroyed.
    ///
    /// This function may return null, such as when called on [`Stdin`],
    /// [`Stdout`], or [`Stderr`] when the console is detached.
    ///
    /// However, borrowing is not strictly required. See [`AsHandle::as_handle`]
    /// for an API which strictly borrows a handle.
    ///
    /// [`Stdin`]: io::Stdin
    /// [`Stdout`]: io::Stdout
    /// [`Stderr`]: io::Stderr
    #[stable(feature = "rust1", since = "1.0.0")]
    fn as_raw_handle(&self) -> RawHandle;
}

/// Constructs I/O objects from raw handles.
#[stable(feature = "from_raw_os", since = "1.1.0")]
pub trait FromRawHandle {
    /// Constructs a new I/O object from the specified raw handle.
    ///
    /// This function is typically used to **consume ownership** of the handle
    /// given, passing responsibility for closing the handle to the returned
    /// object. When used in this way, the returned object
    /// will take responsibility for closing it when the object goes out of
    /// scope.
    ///
    /// However, consuming ownership is not strictly required. Use a
    /// `From<OwnedHandle>::from` implementation for an API which strictly
    /// consumes ownership.
    ///
    /// # Safety
    ///
    /// The `handle` passed in must:
    ///   - be an [owned handle][io-safety]; in particular, it must be open.
    ///   - be a handle for a resource that may be freed via [`CloseHandle`]
    ///     (as opposed to `RegCloseKey` or other close functions).
    ///
    /// Note that the handle *may* have the value `INVALID_HANDLE_VALUE` (-1),
    /// which is sometimes a valid handle value. See [here] for the full story.
    ///
    /// [`CloseHandle`]: https://docs.microsoft.com/en-us/windows/win32/api/handleapi/nf-handleapi-closehandle
    /// [here]: https://devblogs.microsoft.com/oldnewthing/20040302-00/?p=40443
    /// [io-safety]: io#io-safety
    #[stable(feature = "from_raw_os", since = "1.1.0")]
    unsafe fn from_raw_handle(handle: RawHandle) -> Self;
}

/// A trait to express the ability to consume an object and acquire ownership of
/// its raw `HANDLE`.
#[stable(feature = "into_raw_os", since = "1.4.0")]
pub trait IntoRawHandle {
    /// Consumes this object, returning the raw underlying handle.
    ///
    /// This function is typically used to **transfer ownership** of the underlying
    /// handle to the caller. When used in this way, callers are then the unique
    /// owners of the handle and must close it once it's no longer needed.
    ///
    /// However, transferring ownership is not strictly required. Use a
    /// `Into<OwnedHandle>::into` implementation for an API which strictly
    /// transfers ownership.
    #[must_use = "losing the raw handle may leak resources"]
    #[stable(feature = "into_raw_os", since = "1.4.0")]
    fn into_raw_handle(self) -> RawHandle;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRawHandle for fs::File {
    #[inline]
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().as_raw_handle() as RawHandle
    }
}

#[stable(feature = "asraw_stdio", since = "1.21.0")]
impl AsRawHandle for io::Stdin {
    fn as_raw_handle(&self) -> RawHandle {
        stdio_handle(unsafe { sys::c::GetStdHandle(sys::c::STD_INPUT_HANDLE) as RawHandle })
    }
}

#[stable(feature = "asraw_stdio", since = "1.21.0")]
impl AsRawHandle for io::Stdout {
    fn as_raw_handle(&self) -> RawHandle {
        stdio_handle(unsafe { sys::c::GetStdHandle(sys::c::STD_OUTPUT_HANDLE) as RawHandle })
    }
}

#[stable(feature = "asraw_stdio", since = "1.21.0")]
impl AsRawHandle for io::Stderr {
    fn as_raw_handle(&self) -> RawHandle {
        stdio_handle(unsafe { sys::c::GetStdHandle(sys::c::STD_ERROR_HANDLE) as RawHandle })
    }
}

#[stable(feature = "asraw_stdio_locks", since = "1.35.0")]
impl<'a> AsRawHandle for io::StdinLock<'a> {
    fn as_raw_handle(&self) -> RawHandle {
        stdio_handle(unsafe { sys::c::GetStdHandle(sys::c::STD_INPUT_HANDLE) as RawHandle })
    }
}

#[stable(feature = "asraw_stdio_locks", since = "1.35.0")]
impl<'a> AsRawHandle for io::StdoutLock<'a> {
    fn as_raw_handle(&self) -> RawHandle {
        stdio_handle(unsafe { sys::c::GetStdHandle(sys::c::STD_OUTPUT_HANDLE) as RawHandle })
    }
}

#[stable(feature = "asraw_stdio_locks", since = "1.35.0")]
impl<'a> AsRawHandle for io::StderrLock<'a> {
    fn as_raw_handle(&self) -> RawHandle {
        stdio_handle(unsafe { sys::c::GetStdHandle(sys::c::STD_ERROR_HANDLE) as RawHandle })
    }
}

// Translate a handle returned from `GetStdHandle` into a handle to return to
// the user.
fn stdio_handle(raw: RawHandle) -> RawHandle {
    // `GetStdHandle` isn't expected to actually fail, so when it returns
    // `INVALID_HANDLE_VALUE`, it means we were launched from a parent which
    // didn't provide us with stdio handles, such as a parent with a detached
    // console. In that case, return null to the user, which is consistent
    // with what they'd get in the parent, and which avoids the problem that
    // `INVALID_HANDLE_VALUE` aliases the current process handle.
    if raw == sys::c::INVALID_HANDLE_VALUE { ptr::null_mut() } else { raw }
}

#[stable(feature = "from_raw_os", since = "1.1.0")]
impl FromRawHandle for fs::File {
    #[inline]
    unsafe fn from_raw_handle(handle: RawHandle) -> fs::File {
        unsafe {
            let handle = handle as sys::c::HANDLE;
            fs::File::from_inner(sys::fs::File::from_inner(FromInner::from_inner(
                OwnedHandle::from_raw_handle(handle),
            )))
        }
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawHandle for fs::File {
    #[inline]
    fn into_raw_handle(self) -> RawHandle {
        self.into_inner().into_raw_handle() as *mut _
    }
}

/// Extracts raw sockets.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait AsRawSocket {
    /// Extracts the raw socket.
    ///
    /// This function is typically used to **borrow** an owned socket.
    /// When used in this way, this method does **not** pass ownership of the
    /// raw socket to the caller, and the socket is only guaranteed
    /// to be valid while the original object has not yet been destroyed.
    ///
    /// However, borrowing is not strictly required. See [`AsSocket::as_socket`]
    /// for an API which strictly borrows a socket.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn as_raw_socket(&self) -> RawSocket;
}

/// Creates I/O objects from raw sockets.
#[stable(feature = "from_raw_os", since = "1.1.0")]
pub trait FromRawSocket {
    /// Constructs a new I/O object from the specified raw socket.
    ///
    /// This function is typically used to **consume ownership** of the socket
    /// given, passing responsibility for closing the socket to the returned
    /// object. When used in this way, the returned object
    /// will take responsibility for closing it when the object goes out of
    /// scope.
    ///
    /// However, consuming ownership is not strictly required. Use a
    /// `From<OwnedSocket>::from` implementation for an API which strictly
    /// consumes ownership.
    ///
    /// # Safety
    ///
    /// The `socket` passed in must:
    ///   - be an [owned socket][io-safety]; in particular, it must be open.
    ///   - be a socket that may be freed via [`closesocket`].
    ///
    /// [`closesocket`]: https://docs.microsoft.com/en-us/windows/win32/api/winsock2/nf-winsock2-closesocket
    /// [io-safety]: io#io-safety
    #[stable(feature = "from_raw_os", since = "1.1.0")]
    unsafe fn from_raw_socket(sock: RawSocket) -> Self;
}

/// A trait to express the ability to consume an object and acquire ownership of
/// its raw `SOCKET`.
#[stable(feature = "into_raw_os", since = "1.4.0")]
pub trait IntoRawSocket {
    /// Consumes this object, returning the raw underlying socket.
    ///
    /// This function is typically used to **transfer ownership** of the underlying
    /// socket to the caller. When used in this way, callers are then the unique
    /// owners of the socket and must close it once it's no longer needed.
    ///
    /// However, transferring ownership is not strictly required. Use a
    /// `Into<OwnedSocket>::into` implementation for an API which strictly
    /// transfers ownership.
    #[must_use = "losing the raw socket may leak resources"]
    #[stable(feature = "into_raw_os", since = "1.4.0")]
    fn into_raw_socket(self) -> RawSocket;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRawSocket for net::TcpStream {
    #[inline]
    fn as_raw_socket(&self) -> RawSocket {
        self.as_inner().socket().as_raw_socket()
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl AsRawSocket for net::TcpListener {
    #[inline]
    fn as_raw_socket(&self) -> RawSocket {
        self.as_inner().socket().as_raw_socket()
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl AsRawSocket for net::UdpSocket {
    #[inline]
    fn as_raw_socket(&self) -> RawSocket {
        self.as_inner().socket().as_raw_socket()
    }
}

#[stable(feature = "from_raw_os", since = "1.1.0")]
impl FromRawSocket for net::TcpStream {
    #[inline]
    unsafe fn from_raw_socket(sock: RawSocket) -> net::TcpStream {
        unsafe {
            let sock = sys::net::Socket::from_inner(OwnedSocket::from_raw_socket(sock));
            net::TcpStream::from_inner(sys_common::net::TcpStream::from_inner(sock))
        }
    }
}
#[stable(feature = "from_raw_os", since = "1.1.0")]
impl FromRawSocket for net::TcpListener {
    #[inline]
    unsafe fn from_raw_socket(sock: RawSocket) -> net::TcpListener {
        unsafe {
            let sock = sys::net::Socket::from_inner(OwnedSocket::from_raw_socket(sock));
            net::TcpListener::from_inner(sys_common::net::TcpListener::from_inner(sock))
        }
    }
}
#[stable(feature = "from_raw_os", since = "1.1.0")]
impl FromRawSocket for net::UdpSocket {
    #[inline]
    unsafe fn from_raw_socket(sock: RawSocket) -> net::UdpSocket {
        unsafe {
            let sock = sys::net::Socket::from_inner(OwnedSocket::from_raw_socket(sock));
            net::UdpSocket::from_inner(sys_common::net::UdpSocket::from_inner(sock))
        }
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawSocket for net::TcpStream {
    #[inline]
    fn into_raw_socket(self) -> RawSocket {
        self.into_inner().into_socket().into_inner().into_raw_socket()
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawSocket for net::TcpListener {
    #[inline]
    fn into_raw_socket(self) -> RawSocket {
        self.into_inner().into_socket().into_inner().into_raw_socket()
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawSocket for net::UdpSocket {
    #[inline]
    fn into_raw_socket(self) -> RawSocket {
        self.into_inner().into_socket().into_inner().into_raw_socket()
    }
}
