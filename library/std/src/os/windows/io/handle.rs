//! Owned and borrowed OS handles.

#![unstable(feature = "io_safety", issue = "87074")]

use super::raw::{AsRawHandle, FromRawHandle, IntoRawHandle, RawHandle};
use crate::convert::TryFrom;
use crate::ffi::c_void;
use crate::fmt;
use crate::fs;
use crate::marker::PhantomData;
use crate::mem::forget;
use crate::ptr::NonNull;
use crate::sys::c;
use crate::sys_common::{AsInner, FromInner, IntoInner};

/// A borrowed handle.
///
/// This has a lifetime parameter to tie it to the lifetime of something that
/// owns the handle.
///
/// This uses `repr(transparent)` and has the representation of a host handle,
/// so it can be used in FFI in places where a handle is passed as an argument,
/// it is not captured or consumed, and it is never null.
///
/// Note that it *may* have the value `INVALID_HANDLE_VALUE`. See [here] for
/// the full story.
///
/// [here]: https://devblogs.microsoft.com/oldnewthing/20040302-00/?p=40443
#[derive(Copy, Clone)]
#[repr(transparent)]
#[unstable(feature = "io_safety", issue = "87074")]
pub struct BorrowedHandle<'handle> {
    handle: NonNull<c_void>,
    _phantom: PhantomData<&'handle OwnedHandle>,
}

/// An owned handle.
///
/// This closes the handle on drop.
///
/// This uses `repr(transparent)` and has the representation of a host handle,
/// so it can be used in FFI in places where a handle is passed as a consumed
/// argument or returned as an owned value, and is never null.
///
/// Note that it *may* have the value `INVALID_HANDLE_VALUE`. See [here] for
/// the full story. For APIs like `CreateFileW` which report errors with
/// `INVALID_HANDLE_VALUE` instead of null, use [`OptionFileHandle`] instead
/// of `Option<OwnedHandle>`.
///
/// `OwnedHandle` uses [`CloseHandle`] to close its handle on drop. As such,
/// it must not be used with handles to open registry keys which need to be
/// closed with [`RegCloseKey`] instead.
///
/// [`CloseHandle`]: https://docs.microsoft.com/en-us/windows/win32/api/handleapi/nf-handleapi-closehandle
/// [`RegCloseKey`]: https://docs.microsoft.com/en-us/windows/win32/api/winreg/nf-winreg-regclosekey
///
/// [here]: https://devblogs.microsoft.com/oldnewthing/20040302-00/?p=40443
#[repr(transparent)]
#[unstable(feature = "io_safety", issue = "87074")]
pub struct OwnedHandle {
    handle: NonNull<c_void>,
}

/// Similar to `Option<OwnedHandle>`, but intended for use in FFI interfaces
/// where `INVALID_HANDLE_VALUE` is used as the sentry value, and null values
/// are not used at all, such as in the return value of `CreateFileW`.
///
/// The main thing you can do with an `OptionFileHandle` is to convert it into
/// an `OwnedHandle` using its [`TryFrom`] implementation, and this conversion
/// takes care of the check for `INVALID_HANDLE_VALUE`.
///
/// If this holds an owned handle, it closes the handle on drop.
///
/// This uses `repr(transparent)` and has the representation of a host handle,
/// so it can be used in FFI in places where a non-null handle is passed as a
/// consumed argument or returned as an owned value, or it is
/// `INVALID_HANDLE_VALUE` indicating an error or an otherwise absent value.
#[repr(transparent)]
#[unstable(feature = "io_safety", issue = "87074")]
pub struct OptionFileHandle {
    handle: RawHandle,
}

// The Windows [`HANDLE`] type may be transferred across and shared between
// thread boundaries (despite containing a `*mut void`, which in general isn't
// `Send` or `Sync`).
//
// [`HANDLE`]: std::os::windows::raw::HANDLE
unsafe impl Send for OwnedHandle {}
unsafe impl Send for OptionFileHandle {}
unsafe impl Send for BorrowedHandle<'_> {}
unsafe impl Sync for OwnedHandle {}
unsafe impl Sync for OptionFileHandle {}
unsafe impl Sync for BorrowedHandle<'_> {}

impl BorrowedHandle<'_> {
    /// Return a `BorrowedHandle` holding the given raw handle.
    ///
    /// # Safety
    ///
    /// The resource pointed to by `handle` must remain open for the duration
    /// of the returned `BorrowedHandle`, and it must not be null.
    #[inline]
    #[unstable(feature = "io_safety", issue = "87074")]
    pub unsafe fn borrow_raw_handle(handle: RawHandle) -> Self {
        assert!(!handle.is_null());
        Self { handle: NonNull::new_unchecked(handle), _phantom: PhantomData }
    }
}

impl OptionFileHandle {
    /// Return an empty `OptionFileHandle` with no resource.
    #[inline]
    #[unstable(feature = "io_safety", issue = "87074")]
    pub const fn none() -> Self {
        Self { handle: c::INVALID_HANDLE_VALUE }
    }
}

impl TryFrom<OptionFileHandle> for OwnedHandle {
    type Error = ();

    #[inline]
    fn try_from(option: OptionFileHandle) -> Result<Self, ()> {
        let handle = option.handle;
        forget(option);
        if let Some(non_null) = NonNull::new(handle) {
            if non_null.as_ptr() != c::INVALID_HANDLE_VALUE {
                Ok(Self { handle: non_null })
            } else {
                Err(())
            }
        } else {
            // In theory, we ought to be able to assume that the pointer here
            // is never null, change `option.handle` to `NonNull`, and obviate
            // the the panic path here. Unfortunately, Win32 documentation
            // doesn't explicitly guarantee this anywhere.
            //
            // APIs like [`CreateFileW`] itself have `HANDLE` arguments where a
            // null handle indicates an absent value, which wouldn't work if
            // null were a valid handle value, so it seems very unlikely that
            // it could ever return null. But who knows?
            //
            // [`CreateFileW`]: https://docs.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew
            panic!("An `OptionFileHandle` was null!");
        }
    }
}

impl From<OwnedHandle> for OptionFileHandle {
    #[inline]
    fn from(owned: OwnedHandle) -> Self {
        let handle = owned.handle;
        forget(owned);
        Self { handle: handle.as_ptr() }
    }
}

impl AsRawHandle for BorrowedHandle<'_> {
    #[inline]
    fn as_raw_handle(&self) -> RawHandle {
        self.handle.as_ptr()
    }
}

impl AsRawHandle for OwnedHandle {
    #[inline]
    fn as_raw_handle(&self) -> RawHandle {
        self.handle.as_ptr()
    }
}

impl IntoRawHandle for OwnedHandle {
    #[inline]
    fn into_raw_handle(self) -> RawHandle {
        let handle = self.handle.as_ptr();
        forget(self);
        handle
    }
}

impl FromRawHandle for OwnedHandle {
    /// Constructs a new instance of `Self` from the given raw handle.
    ///
    /// # Safety
    ///
    /// The resource pointed to by `handle` must be open and suitable for
    /// assuming ownership. The resource must not require any cleanup other
    /// than `CloseHandle`.
    ///
    /// In particular, it must not be used with handles to open registry
    /// keys which need to be closed with [`RegCloseKey`] instead.
    ///
    /// [`RegCloseKey`]: https://docs.microsoft.com/en-us/windows/win32/api/winreg/nf-winreg-regclosekey
    #[inline]
    unsafe fn from_raw_handle(handle: RawHandle) -> Self {
        assert!(!handle.is_null());
        Self { handle: NonNull::new_unchecked(handle) }
    }
}

impl FromRawHandle for OptionFileHandle {
    /// Constructs a new instance of `Self` from the given raw handle.
    ///
    /// # Safety
    ///
    /// The resource pointed to by `handle` must be either open and otherwise
    /// unowned, or equal to `INVALID_HANDLE_VALUE``. Note that not all Windows
    /// APIs use `INVALID_HANDLE_VALUE` for errors; see [here] for the full
    /// story.
    ///
    /// [here]: https://devblogs.microsoft.com/oldnewthing/20040302-00/?p=40443
    #[inline]
    unsafe fn from_raw_handle(handle: RawHandle) -> Self {
        assert!(!handle.is_null());
        Self { handle }
    }
}

impl Drop for OwnedHandle {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let _ = c::CloseHandle(self.handle.as_ptr());
        }
    }
}

impl Drop for OptionFileHandle {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let _ = c::CloseHandle(self.handle);
        }
    }
}

impl fmt::Debug for BorrowedHandle<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BorrowedHandle").field("handle", &self.handle).finish()
    }
}

impl fmt::Debug for OwnedHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnedHandle").field("handle", &self.handle).finish()
    }
}

impl fmt::Debug for OptionFileHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OptionFileHandle").field("handle", &self.handle).finish()
    }
}

/// A trait to borrow the handle from an underlying object.
#[unstable(feature = "io_safety", issue = "87074")]
pub trait AsHandle {
    /// Borrows the handle.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::fs::File;
    /// # use std::io;
    /// use std::os::windows::{AsHandle, BorrowedHandle};
    ///
    /// let mut f = File::open("foo.txt")?;
    /// let borrowed_handle: BorrowedHandle<'_> = f.as_handle();
    /// # Ok::<(), io::Error>(())
    /// ```
    fn as_handle(&self) -> BorrowedHandle<'_>;
}

impl AsHandle for BorrowedHandle<'_> {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        *self
    }
}

impl AsHandle for OwnedHandle {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        // Safety: `OwnedHandle` and `BorrowedHandle` have the same validity
        // invariants, and the `BorrowdHandle` is bounded by the lifetime
        // of `&self`.
        unsafe { BorrowedHandle::borrow_raw_handle(self.as_raw_handle()) }
    }
}

impl AsHandle for fs::File {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        self.as_inner().as_handle()
    }
}

impl From<fs::File> for OwnedHandle {
    #[inline]
    fn from(file: fs::File) -> OwnedHandle {
        file.into_inner().into_inner().into_inner().into()
    }
}

impl From<OwnedHandle> for fs::File {
    #[inline]
    fn from(owned: OwnedHandle) -> Self {
        Self::from_inner(FromInner::from_inner(FromInner::from_inner(owned)))
    }
}

impl AsHandle for crate::io::Stdin {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        unsafe { BorrowedHandle::borrow_raw_handle(self.as_raw_handle()) }
    }
}

impl<'a> AsHandle for crate::io::StdinLock<'a> {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        unsafe { BorrowedHandle::borrow_raw_handle(self.as_raw_handle()) }
    }
}

impl AsHandle for crate::io::Stdout {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        unsafe { BorrowedHandle::borrow_raw_handle(self.as_raw_handle()) }
    }
}

impl<'a> AsHandle for crate::io::StdoutLock<'a> {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        unsafe { BorrowedHandle::borrow_raw_handle(self.as_raw_handle()) }
    }
}

impl AsHandle for crate::io::Stderr {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        unsafe { BorrowedHandle::borrow_raw_handle(self.as_raw_handle()) }
    }
}

impl<'a> AsHandle for crate::io::StderrLock<'a> {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        unsafe { BorrowedHandle::borrow_raw_handle(self.as_raw_handle()) }
    }
}

impl AsHandle for crate::process::ChildStdin {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        unsafe { BorrowedHandle::borrow_raw_handle(self.as_raw_handle()) }
    }
}

impl From<crate::process::ChildStdin> for OwnedHandle {
    #[inline]
    fn from(child_stdin: crate::process::ChildStdin) -> OwnedHandle {
        unsafe { OwnedHandle::from_raw_handle(child_stdin.into_raw_handle()) }
    }
}

impl AsHandle for crate::process::ChildStdout {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        unsafe { BorrowedHandle::borrow_raw_handle(self.as_raw_handle()) }
    }
}

impl From<crate::process::ChildStdout> for OwnedHandle {
    #[inline]
    fn from(child_stdout: crate::process::ChildStdout) -> OwnedHandle {
        unsafe { OwnedHandle::from_raw_handle(child_stdout.into_raw_handle()) }
    }
}

impl AsHandle for crate::process::ChildStderr {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        unsafe { BorrowedHandle::borrow_raw_handle(self.as_raw_handle()) }
    }
}

impl From<crate::process::ChildStderr> for OwnedHandle {
    #[inline]
    fn from(child_stderr: crate::process::ChildStderr) -> OwnedHandle {
        unsafe { OwnedHandle::from_raw_handle(child_stderr.into_raw_handle()) }
    }
}

impl<T> AsHandle for crate::thread::JoinHandle<T> {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        unsafe { BorrowedHandle::borrow_raw_handle(self.as_raw_handle()) }
    }
}

impl<T> From<crate::thread::JoinHandle<T>> for OwnedHandle {
    #[inline]
    fn from(join_handle: crate::thread::JoinHandle<T>) -> OwnedHandle {
        join_handle.into_inner().into_handle().into_inner()
    }
}
