//! SOLID-specific extensions to general I/O primitives
//!
//! Just like raw pointers, raw SOLID Sockets file descriptors point to
//! resources with dynamic lifetimes, and they can dangle if they outlive their
//! resources or be forged if they're created from invalid values.
//!
//! This module provides three types for representing raw file descriptors
//! with different ownership properties: raw, borrowed, and owned, which are
//! analogous to types used for representing pointers:
//!
//! | Type               | Analogous to |
//! | ------------------ | ------------ |
//! | [`RawFd`]          | `*const _`   |
//! | [`BorrowedFd<'a>`] | `&'a _`      |
//! | [`OwnedFd`]        | `Box<_>`     |
//!
//! Like raw pointers, `RawFd` values are primitive values. And in new code,
//! they should be considered unsafe to do I/O on (analogous to dereferencing
//! them). Rust did not always provide this guidance, so existing code in the
//! Rust ecosystem often doesn't mark `RawFd` usage as unsafe. Once the
//! `io_safety` feature is stable, libraries will be encouraged to migrate,
//! either by adding `unsafe` to APIs that dereference `RawFd` values, or by
//! using to `BorrowedFd` or `OwnedFd` instead.
//!
//! Like references, `BorrowedFd` values are tied to a lifetime, to ensure
//! that they don't outlive the resource they point to. These are safe to
//! use. `BorrowedFd` values may be used in APIs which provide safe access to
//! any system call except for:
//!
//!  - `close`, because that would end the dynamic lifetime of the resource
//!    without ending the lifetime of the file descriptor.
//!
//!  - `dup2`/`dup3`, in the second argument, because this argument is
//!    closed and assigned a new resource, which may break the assumptions
//!    other code using that file descriptor.
//!
//! `BorrowedFd` values may be used in APIs which provide safe access to `dup`
//! system calls, so types implementing `AsFd` or `From<OwnedFd>` should not
//! assume they always have exclusive access to the underlying file
//! description.
//!
//! Like boxes, `OwnedFd` values conceptually own the resource they point to,
//! and free (close) it when they are dropped.
//!
//! [`BorrowedFd<'a>`]: crate::os::solid::io::BorrowedFd

#![unstable(feature = "solid_ext", issue = "none")]

use crate::marker::PhantomData;
use crate::mem::ManuallyDrop;
use crate::sys_common::{AsInner, FromInner, IntoInner};
use crate::{fmt, net, sys};

/// Raw file descriptors.
pub type RawFd = i32;

// The max of this is -2, in two's complement. -1 is `SOLID_NET_INVALID_FD`.
type ValidRawFd = core::num::niche_types::NotAllOnes<RawFd>;

/// A borrowed SOLID Sockets file descriptor.
///
/// This has a lifetime parameter to tie it to the lifetime of something that
/// owns the socket.
///
/// This uses `repr(transparent)` and has the representation of a host file
/// descriptor, so it can be used in FFI in places where a socket is passed as
/// an argument, it is not captured or consumed, and it never has the value
/// `SOLID_NET_INVALID_FD`.
///
/// This type's `.to_owned()` implementation returns another `BorrowedFd`
/// rather than an `OwnedFd`. It just makes a trivial copy of the raw
/// socket, which is then borrowed under the same lifetime.
#[derive(Copy, Clone)]
#[repr(transparent)]
#[rustc_nonnull_optimization_guaranteed]
pub struct BorrowedFd<'socket> {
    fd: ValidRawFd,
    _phantom: PhantomData<&'socket OwnedFd>,
}

/// An owned SOLID Sockets file descriptor.
///
/// This closes the file descriptor on drop.
///
/// This uses `repr(transparent)` and has the representation of a host file
/// descriptor, so it can be used in FFI in places where a socket is passed as
/// an argument, it is not captured or consumed, and it never has the value
/// `SOLID_NET_INVALID_FD`.
#[repr(transparent)]
#[rustc_nonnull_optimization_guaranteed]
pub struct OwnedFd {
    fd: ValidRawFd,
}

impl BorrowedFd<'_> {
    /// Returns a `BorrowedFd` holding the given raw file descriptor.
    ///
    /// # Safety
    ///
    /// The resource pointed to by `fd` must remain open for the duration of
    /// the returned `BorrowedFd`, and it must not have the value
    /// `SOLID_NET_INVALID_FD`.
    #[inline]
    pub const unsafe fn borrow_raw(fd: RawFd) -> Self {
        assert!(fd != -1 as RawFd);
        // SAFETY: we just asserted that the value is in the valid range and
        // isn't `-1` (the only value bigger than `0xFF_FF_FF_FE` unsigned)
        let fd = unsafe { ValidRawFd::new_unchecked(fd) };
        Self { fd, _phantom: PhantomData }
    }
}

impl OwnedFd {
    /// Creates a new `OwnedFd` instance that shares the same underlying file
    /// description as the existing `OwnedFd` instance.
    pub fn try_clone(&self) -> crate::io::Result<Self> {
        self.as_fd().try_clone_to_owned()
    }
}

impl BorrowedFd<'_> {
    /// Creates a new `OwnedFd` instance that shares the same underlying file
    /// description as the existing `BorrowedFd` instance.
    pub fn try_clone_to_owned(&self) -> crate::io::Result<OwnedFd> {
        let fd = sys::net::cvt(unsafe { crate::sys::abi::sockets::dup(self.as_raw_fd()) })?;
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }
}

impl AsRawFd for BorrowedFd<'_> {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.fd.as_inner()
    }
}

impl AsRawFd for OwnedFd {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.fd.as_inner()
    }
}

impl IntoRawFd for OwnedFd {
    #[inline]
    fn into_raw_fd(self) -> RawFd {
        ManuallyDrop::new(self).fd.as_inner()
    }
}

impl FromRawFd for OwnedFd {
    /// Constructs a new instance of `Self` from the given raw file descriptor.
    ///
    /// # Safety
    ///
    /// The resource pointed to by `fd` must be open and suitable for assuming
    /// ownership. The resource must not require any cleanup other than `close`.
    #[inline]
    unsafe fn from_raw_fd(fd: RawFd) -> Self {
        assert_ne!(fd, -1 as RawFd);
        // SAFETY: we just asserted that the value is in the valid range and
        // isn't `-1` (the only value bigger than `0xFF_FF_FF_FE` unsigned)
        let fd = unsafe { ValidRawFd::new_unchecked(fd) };
        Self { fd }
    }
}

impl Drop for OwnedFd {
    #[inline]
    fn drop(&mut self) {
        unsafe { crate::sys::abi::sockets::close(self.fd.as_inner()) };
    }
}

impl fmt::Debug for BorrowedFd<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BorrowedFd").field("fd", &self.fd).finish()
    }
}

impl fmt::Debug for OwnedFd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnedFd").field("fd", &self.fd).finish()
    }
}

macro_rules! impl_is_terminal {
    ($($t:ty),*$(,)?) => {$(
        #[unstable(feature = "sealed", issue = "none")]
        impl crate::sealed::Sealed for $t {}

        #[stable(feature = "is_terminal", since = "1.70.0")]
        impl crate::io::IsTerminal for $t {
            #[inline]
            fn is_terminal(&self) -> bool {
                crate::sys::io::is_terminal(self)
            }
        }
    )*}
}

impl_is_terminal!(BorrowedFd<'_>, OwnedFd);

/// A trait to borrow the SOLID Sockets file descriptor from an underlying
/// object.
pub trait AsFd {
    /// Borrows the file descriptor.
    fn as_fd(&self) -> BorrowedFd<'_>;
}

impl<T: AsFd> AsFd for &T {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        T::as_fd(self)
    }
}

impl<T: AsFd> AsFd for &mut T {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        T::as_fd(self)
    }
}

impl AsFd for BorrowedFd<'_> {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        *self
    }
}

impl AsFd for OwnedFd {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        // Safety: `OwnedFd` and `BorrowedFd` have the same validity
        // invariants, and the `BorrowedFd` is bounded by the lifetime
        // of `&self`.
        unsafe { BorrowedFd::borrow_raw(self.as_raw_fd()) }
    }
}

macro_rules! impl_owned_fd_traits {
    ($($t:ident)*) => {$(
        impl AsFd for net::$t {
            #[inline]
            fn as_fd(&self) -> BorrowedFd<'_> {
                self.as_inner().socket().as_fd()
            }
        }

        impl From<net::$t> for OwnedFd {
            #[inline]
            fn from(socket: net::$t) -> OwnedFd {
                socket.into_inner().into_socket().into_inner()
            }
        }

        impl From<OwnedFd> for net::$t {
            #[inline]
            fn from(owned_fd: OwnedFd) -> Self {
                Self::from_inner(FromInner::from_inner(FromInner::from_inner(owned_fd)))
            }
        }
    )*};
}
impl_owned_fd_traits! { TcpStream TcpListener UdpSocket }

/// This impl allows implementing traits that require `AsFd` on Arc.
/// ```
/// # #[cfg(target_os = "solid_asp3")] mod group_cfg {
/// # use std::os::solid::io::AsFd;
/// use std::net::UdpSocket;
/// use std::sync::Arc;
///
/// trait MyTrait: AsFd {}
/// impl MyTrait for Arc<UdpSocket> {}
/// impl MyTrait for Box<UdpSocket> {}
/// # }
/// ```
impl<T: AsFd> AsFd for crate::sync::Arc<T> {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        (**self).as_fd()
    }
}

impl<T: AsFd> AsFd for crate::rc::Rc<T> {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        (**self).as_fd()
    }
}

impl<T: AsFd> AsFd for Box<T> {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        (**self).as_fd()
    }
}

/// A trait to extract the raw SOLID Sockets file descriptor from an underlying
/// object.
pub trait AsRawFd {
    /// Extracts the raw file descriptor.
    ///
    /// This method does **not** pass ownership of the raw file descriptor
    /// to the caller. The descriptor is only guaranteed to be valid while
    /// the original object has not yet been destroyed.
    fn as_raw_fd(&self) -> RawFd;
}

/// A trait to express the ability to construct an object from a raw file
/// descriptor.
pub trait FromRawFd {
    /// Constructs a new instance of `Self` from the given raw file
    /// descriptor.
    ///
    /// This function is typically used to **consume ownership** of the
    /// specified file descriptor. When used in this way, the returned object
    /// will take responsibility for closing it when the object goes out of
    /// scope.
    ///
    /// However, consuming ownership is not strictly required. Use a
    /// [`From<OwnedFd>::from`] implementation for an API which strictly
    /// consumes ownership.
    ///
    /// # Safety
    ///
    /// The `fd` passed in must be an [owned file descriptor][io-safety];
    /// in particular, it must be open.
    ///
    /// [io-safety]: io#io-safety
    unsafe fn from_raw_fd(fd: RawFd) -> Self;
}

/// A trait to express the ability to consume an object and acquire ownership of
/// its raw file descriptor.
pub trait IntoRawFd {
    /// Consumes this object, returning the raw underlying file descriptor.
    ///
    /// This function **transfers ownership** of the underlying file descriptor
    /// to the caller. Callers are then the unique owners of the file descriptor
    /// and must close the descriptor once it's no longer needed.
    #[must_use = "losing the raw file descriptor may leak resources"]
    fn into_raw_fd(self) -> RawFd;
}

#[stable(feature = "raw_fd_reflexive_traits", since = "1.48.0")]
impl AsRawFd for RawFd {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        *self
    }
}
#[stable(feature = "raw_fd_reflexive_traits", since = "1.48.0")]
impl IntoRawFd for RawFd {
    #[inline]
    fn into_raw_fd(self) -> RawFd {
        self
    }
}
#[stable(feature = "raw_fd_reflexive_traits", since = "1.48.0")]
impl FromRawFd for RawFd {
    #[inline]
    unsafe fn from_raw_fd(fd: RawFd) -> RawFd {
        fd
    }
}

macro_rules! impl_as_raw_fd {
    ($($t:ident)*) => {$(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl AsRawFd for net::$t {
            #[inline]
            fn as_raw_fd(&self) -> RawFd {
                self.as_inner().socket().as_raw_fd()
            }
        }
    )*};
}
impl_as_raw_fd! { TcpStream TcpListener UdpSocket }

macro_rules! impl_from_raw_fd {
    ($($t:ident)*) => {$(
        #[stable(feature = "from_raw_os", since = "1.1.0")]
        impl FromRawFd for net::$t {
            #[inline]
            unsafe fn from_raw_fd(fd: RawFd) -> net::$t {
                let socket = unsafe { sys::net::Socket::from_raw_fd(fd) };
                net::$t::from_inner(sys::net::$t::from_inner(socket))
            }
        }
    )*};
}
impl_from_raw_fd! { TcpStream TcpListener UdpSocket }

macro_rules! impl_into_raw_fd {
    ($($t:ident)*) => {$(
        #[stable(feature = "into_raw_os", since = "1.4.0")]
        impl IntoRawFd for net::$t {
            #[inline]
            fn into_raw_fd(self) -> RawFd {
                self.into_inner().into_socket().into_raw_fd()
            }
        }
    )*};
}
impl_into_raw_fd! { TcpStream TcpListener UdpSocket }
