//! Owned and borrowed OS sockets.

#![stable(feature = "io_safety", since = "1.63.0")]

use super::raw::{AsRawSocket, FromRawSocket, IntoRawSocket, RawSocket};
use crate::marker::PhantomData;
use crate::mem::{self, ManuallyDrop};
#[cfg(not(target_vendor = "uwp"))]
use crate::sys::cvt;
use crate::{fmt, io, sys};

/// A borrowed socket.
///
/// This has a lifetime parameter to tie it to the lifetime of something that
/// owns the socket.
///
/// This uses `repr(transparent)` and has the representation of a host socket,
/// so it can be used in FFI in places where a socket is passed as an argument,
/// it is not captured or consumed, and it never has the value
/// `INVALID_SOCKET`.
///
/// This type's `.to_owned()` implementation returns another `BorrowedSocket`
/// rather than an `OwnedSocket`. It just makes a trivial copy of the raw
/// socket, which is then borrowed under the same lifetime.
#[derive(Copy, Clone)]
#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(0)]
// This is -2, in two's complement. -1 is `INVALID_SOCKET`.
#[cfg_attr(target_pointer_width = "32", rustc_layout_scalar_valid_range_end(0xFF_FF_FF_FE))]
#[cfg_attr(
    target_pointer_width = "64",
    rustc_layout_scalar_valid_range_end(0xFF_FF_FF_FF_FF_FF_FF_FE)
)]
#[rustc_nonnull_optimization_guaranteed]
#[stable(feature = "io_safety", since = "1.63.0")]
pub struct BorrowedSocket<'socket> {
    socket: RawSocket,
    _phantom: PhantomData<&'socket OwnedSocket>,
}

/// An owned socket.
///
/// This closes the socket on drop.
///
/// This uses `repr(transparent)` and has the representation of a host socket,
/// so it can be used in FFI in places where a socket is passed as a consumed
/// argument or returned as an owned value, and it never has the value
/// `INVALID_SOCKET`.
#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(0)]
// This is -2, in two's complement. -1 is `INVALID_SOCKET`.
#[cfg_attr(target_pointer_width = "32", rustc_layout_scalar_valid_range_end(0xFF_FF_FF_FE))]
#[cfg_attr(
    target_pointer_width = "64",
    rustc_layout_scalar_valid_range_end(0xFF_FF_FF_FF_FF_FF_FF_FE)
)]
#[rustc_nonnull_optimization_guaranteed]
#[stable(feature = "io_safety", since = "1.63.0")]
pub struct OwnedSocket {
    socket: RawSocket,
}

impl BorrowedSocket<'_> {
    /// Returns a `BorrowedSocket` holding the given raw socket.
    ///
    /// # Safety
    ///
    /// The resource pointed to by `raw` must remain open for the duration of
    /// the returned `BorrowedSocket`, and it must not have the value
    /// `INVALID_SOCKET`.
    #[inline]
    #[rustc_const_stable(feature = "io_safety", since = "1.63.0")]
    #[stable(feature = "io_safety", since = "1.63.0")]
    pub const unsafe fn borrow_raw(socket: RawSocket) -> Self {
        assert!(socket != sys::c::INVALID_SOCKET as RawSocket);
        unsafe { Self { socket, _phantom: PhantomData } }
    }
}

impl OwnedSocket {
    /// Creates a new `OwnedSocket` instance that shares the same underlying
    /// object as the existing `OwnedSocket` instance.
    #[stable(feature = "io_safety", since = "1.63.0")]
    pub fn try_clone(&self) -> io::Result<Self> {
        self.as_socket().try_clone_to_owned()
    }

    // FIXME(strict_provenance_magic): we defined RawSocket to be a u64 ;-;
    #[allow(fuzzy_provenance_casts)]
    #[cfg(not(target_vendor = "uwp"))]
    pub(crate) fn set_no_inherit(&self) -> io::Result<()> {
        cvt(unsafe {
            sys::c::SetHandleInformation(
                self.as_raw_socket() as sys::c::HANDLE,
                sys::c::HANDLE_FLAG_INHERIT,
                0,
            )
        })
        .map(drop)
    }

    #[cfg(target_vendor = "uwp")]
    pub(crate) fn set_no_inherit(&self) -> io::Result<()> {
        Err(io::const_io_error!(io::ErrorKind::Unsupported, "Unavailable on UWP"))
    }
}

impl BorrowedSocket<'_> {
    /// Creates a new `OwnedSocket` instance that shares the same underlying
    /// object as the existing `BorrowedSocket` instance.
    #[stable(feature = "io_safety", since = "1.63.0")]
    pub fn try_clone_to_owned(&self) -> io::Result<OwnedSocket> {
        let mut info = unsafe { mem::zeroed::<sys::c::WSAPROTOCOL_INFOW>() };
        let result = unsafe {
            sys::c::WSADuplicateSocketW(
                self.as_raw_socket() as sys::c::SOCKET,
                sys::c::GetCurrentProcessId(),
                &mut info,
            )
        };
        sys::net::cvt(result)?;
        let socket = unsafe {
            sys::c::WSASocketW(
                info.iAddressFamily,
                info.iSocketType,
                info.iProtocol,
                &info,
                0,
                sys::c::WSA_FLAG_OVERLAPPED | sys::c::WSA_FLAG_NO_HANDLE_INHERIT,
            )
        };

        if socket != sys::c::INVALID_SOCKET {
            unsafe { Ok(OwnedSocket::from_raw_socket(socket as RawSocket)) }
        } else {
            let error = unsafe { sys::c::WSAGetLastError() };

            if error != sys::c::WSAEPROTOTYPE && error != sys::c::WSAEINVAL {
                return Err(io::Error::from_raw_os_error(error));
            }

            let socket = unsafe {
                sys::c::WSASocketW(
                    info.iAddressFamily,
                    info.iSocketType,
                    info.iProtocol,
                    &info,
                    0,
                    sys::c::WSA_FLAG_OVERLAPPED,
                )
            };

            if socket == sys::c::INVALID_SOCKET {
                return Err(last_error());
            }

            unsafe {
                let socket = OwnedSocket::from_raw_socket(socket as RawSocket);
                socket.set_no_inherit()?;
                Ok(socket)
            }
        }
    }
}

/// Returns the last error from the Windows socket interface.
fn last_error() -> io::Error {
    io::Error::from_raw_os_error(unsafe { sys::c::WSAGetLastError() })
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsRawSocket for BorrowedSocket<'_> {
    #[inline]
    fn as_raw_socket(&self) -> RawSocket {
        self.socket
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsRawSocket for OwnedSocket {
    #[inline]
    fn as_raw_socket(&self) -> RawSocket {
        self.socket
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl IntoRawSocket for OwnedSocket {
    #[inline]
    fn into_raw_socket(self) -> RawSocket {
        ManuallyDrop::new(self).socket
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl FromRawSocket for OwnedSocket {
    #[inline]
    unsafe fn from_raw_socket(socket: RawSocket) -> Self {
        unsafe {
            debug_assert_ne!(socket, sys::c::INVALID_SOCKET as RawSocket);
            Self { socket }
        }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl Drop for OwnedSocket {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let _ = sys::c::closesocket(self.socket as sys::c::SOCKET);
        }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl fmt::Debug for BorrowedSocket<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BorrowedSocket").field("socket", &self.socket).finish()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl fmt::Debug for OwnedSocket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnedSocket").field("socket", &self.socket).finish()
    }
}

/// A trait to borrow the socket from an underlying object.
#[stable(feature = "io_safety", since = "1.63.0")]
pub trait AsSocket {
    /// Borrows the socket.
    #[stable(feature = "io_safety", since = "1.63.0")]
    fn as_socket(&self) -> BorrowedSocket<'_>;
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl<T: AsSocket> AsSocket for &T {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        T::as_socket(self)
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl<T: AsSocket> AsSocket for &mut T {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        T::as_socket(self)
    }
}

#[stable(feature = "as_windows_ptrs", since = "1.71.0")]
/// This impl allows implementing traits that require `AsSocket` on Arc.
/// ```
/// # #[cfg(windows)] mod group_cfg {
/// # use std::os::windows::io::AsSocket;
/// use std::net::UdpSocket;
/// use std::sync::Arc;
///
/// trait MyTrait: AsSocket {}
/// impl MyTrait for Arc<UdpSocket> {}
/// impl MyTrait for Box<UdpSocket> {}
/// # }
/// ```
impl<T: AsSocket> AsSocket for crate::sync::Arc<T> {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        (**self).as_socket()
    }
}

#[stable(feature = "as_windows_ptrs", since = "1.71.0")]
impl<T: AsSocket> AsSocket for crate::rc::Rc<T> {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        (**self).as_socket()
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: AsSocket + ?Sized> AsSocket for crate::rc::UniqueRc<T> {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        (**self).as_socket()
    }
}

#[stable(feature = "as_windows_ptrs", since = "1.71.0")]
impl<T: AsSocket> AsSocket for Box<T> {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        (**self).as_socket()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsSocket for BorrowedSocket<'_> {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        *self
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsSocket for OwnedSocket {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        // Safety: `OwnedSocket` and `BorrowedSocket` have the same validity
        // invariants, and the `BorrowedSocket` is bounded by the lifetime
        // of `&self`.
        unsafe { BorrowedSocket::borrow_raw(self.as_raw_socket()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsSocket for crate::net::TcpStream {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        unsafe { BorrowedSocket::borrow_raw(self.as_raw_socket()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<crate::net::TcpStream> for OwnedSocket {
    /// Takes ownership of a [`TcpStream`](crate::net::TcpStream)'s socket.
    #[inline]
    fn from(tcp_stream: crate::net::TcpStream) -> OwnedSocket {
        unsafe { OwnedSocket::from_raw_socket(tcp_stream.into_raw_socket()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<OwnedSocket> for crate::net::TcpStream {
    #[inline]
    fn from(owned: OwnedSocket) -> Self {
        unsafe { Self::from_raw_socket(owned.into_raw_socket()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsSocket for crate::net::TcpListener {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        unsafe { BorrowedSocket::borrow_raw(self.as_raw_socket()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<crate::net::TcpListener> for OwnedSocket {
    /// Takes ownership of a [`TcpListener`](crate::net::TcpListener)'s socket.
    #[inline]
    fn from(tcp_listener: crate::net::TcpListener) -> OwnedSocket {
        unsafe { OwnedSocket::from_raw_socket(tcp_listener.into_raw_socket()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<OwnedSocket> for crate::net::TcpListener {
    #[inline]
    fn from(owned: OwnedSocket) -> Self {
        unsafe { Self::from_raw_socket(owned.into_raw_socket()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsSocket for crate::net::UdpSocket {
    #[inline]
    fn as_socket(&self) -> BorrowedSocket<'_> {
        unsafe { BorrowedSocket::borrow_raw(self.as_raw_socket()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<crate::net::UdpSocket> for OwnedSocket {
    /// Takes ownership of a [`UdpSocket`](crate::net::UdpSocket)'s underlying socket.
    #[inline]
    fn from(udp_socket: crate::net::UdpSocket) -> OwnedSocket {
        unsafe { OwnedSocket::from_raw_socket(udp_socket.into_raw_socket()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<OwnedSocket> for crate::net::UdpSocket {
    #[inline]
    fn from(owned: OwnedSocket) -> Self {
        unsafe { Self::from_raw_socket(owned.into_raw_socket()) }
    }
}
