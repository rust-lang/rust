//! Owned and borrowed OS handles.

#![unstable(feature = "io_safety", issue = "87074")]

use super::raw::{AsRawHandle, FromRawHandle, IntoRawHandle, RawHandle};
use crate::convert::TryFrom;
use crate::fmt;
use crate::fs;
use crate::io;
use crate::marker::PhantomData;
use crate::mem::forget;
use crate::sys::c;
use crate::sys::cvt;
use crate::sys_common::{AsInner, FromInner, IntoInner};

/// A borrowed handle.
///
/// This has a lifetime parameter to tie it to the lifetime of something that
/// owns the handle.
///
/// This uses `repr(transparent)` and has the representation of a host handle,
/// so it can be used in FFI in places where a handle is passed as an argument,
/// it is not captured or consumed.
///
/// Note that it *may* have the value `INVALID_HANDLE_VALUE` (-1), which is
/// sometimes a valid handle value. See [here] for the full story.
///
/// And, it *may* have the value `NULL` (0), which can occur when consoles are
/// detached from processes, or when `windows_subsystem` is used.
///
/// [here]: https://devblogs.microsoft.com/oldnewthing/20040302-00/?p=40443
#[derive(Copy, Clone)]
#[repr(transparent)]
#[unstable(feature = "io_safety", issue = "87074")]
pub struct BorrowedHandle<'handle> {
    handle: RawHandle,
    _phantom: PhantomData<&'handle OwnedHandle>,
}

/// An owned handle.
///
/// This closes the handle on drop.
///
/// Note that it *may* have the value `INVALID_HANDLE_VALUE` (-1), which is
/// sometimes a valid handle value. See [here] for the full story.
///
/// And, it *may* have the value `NULL` (0), which can occur when consoles are
/// detached from processes, or when `windows_subsystem` is used.
///
/// `OwnedHandle` uses [`CloseHandle`] to close its handle on drop. As such,
/// it must not be used with handles to open registry keys which need to be
/// closed with [`RegCloseKey`] instead.
///
/// [`CloseHandle`]: https://docs.microsoft.com/en-us/windows/win32/api/handleapi/nf-handleapi-closehandle
/// [`RegCloseKey`]: https://docs.microsoft.com/en-us/windows/win32/api/winreg/nf-winreg-regclosekey
///
/// [here]: https://devblogs.microsoft.com/oldnewthing/20040302-00/?p=40443
#[unstable(feature = "io_safety", issue = "87074")]
pub struct OwnedHandle {
    handle: RawHandle,
}

/// FFI type for handles in return values or out parameters, where `NULL` is used
/// as a sentry value to indicate errors, such as in the return value of `CreateThread`. This uses
/// `repr(transparent)` and has the representation of a host handle, so that it can be used in such
/// FFI declarations.
///
/// The only thing you can usefully do with a `HandleOrNull` is to convert it into an
/// `OwnedHandle` using its [`TryFrom`] implementation; this conversion takes care of the check for
/// `NULL`. This ensures that such FFI calls cannot start using the handle without
/// checking for `NULL` first.
///
/// This type concerns any value other than `NULL` to be valid, including `INVALID_HANDLE_VALUE`.
/// This is because APIs that use `NULL` as their sentry value don't treat `INVALID_HANDLE_VALUE`
/// as special.
///
/// If this holds a valid handle, it will close the handle on drop.
#[repr(transparent)]
#[unstable(feature = "io_safety", issue = "87074")]
#[derive(Debug)]
pub struct HandleOrNull(OwnedHandle);

/// FFI type for handles in return values or out parameters, where `INVALID_HANDLE_VALUE` is used
/// as a sentry value to indicate errors, such as in the return value of `CreateFileW`. This uses
/// `repr(transparent)` and has the representation of a host handle, so that it can be used in such
/// FFI declarations.
///
/// The only thing you can usefully do with a `HandleOrInvalid` is to convert it into an
/// `OwnedHandle` using its [`TryFrom`] implementation; this conversion takes care of the check for
/// `INVALID_HANDLE_VALUE`. This ensures that such FFI calls cannot start using the handle without
/// checking for `INVALID_HANDLE_VALUE` first.
///
/// This type concerns any value other than `INVALID_HANDLE_VALUE` to be valid, including `NULL`.
/// This is because APIs that use `INVALID_HANDLE_VALUE` as their sentry value may return `NULL`
/// under `windows_subsystem = "windows"` or other situations where I/O devices are detached.
///
/// If this holds a valid handle, it will close the handle on drop.
#[repr(transparent)]
#[unstable(feature = "io_safety", issue = "87074")]
#[derive(Debug)]
pub struct HandleOrInvalid(OwnedHandle);

// The Windows [`HANDLE`] type may be transferred across and shared between
// thread boundaries (despite containing a `*mut void`, which in general isn't
// `Send` or `Sync`).
//
// [`HANDLE`]: std::os::windows::raw::HANDLE
unsafe impl Send for OwnedHandle {}
unsafe impl Send for HandleOrNull {}
unsafe impl Send for HandleOrInvalid {}
unsafe impl Send for BorrowedHandle<'_> {}
unsafe impl Sync for OwnedHandle {}
unsafe impl Sync for HandleOrNull {}
unsafe impl Sync for HandleOrInvalid {}
unsafe impl Sync for BorrowedHandle<'_> {}

impl BorrowedHandle<'_> {
    /// Return a `BorrowedHandle` holding the given raw handle.
    ///
    /// # Safety
    ///
    /// The resource pointed to by `handle` must be a valid open handle, it
    /// must remain open for the duration of the returned `BorrowedHandle`.
    ///
    /// Note that it *may* have the value `INVALID_HANDLE_VALUE` (-1), which is
    /// sometimes a valid handle value. See [here] for the full story.
    ///
    /// And, it *may* have the value `NULL` (0), which can occur when consoles are
    /// detached from processes, or when `windows_subsystem` is used.
    ///
    /// [here]: https://devblogs.microsoft.com/oldnewthing/20040302-00/?p=40443
    #[inline]
    #[unstable(feature = "io_safety", issue = "87074")]
    pub unsafe fn borrow_raw_handle(handle: RawHandle) -> Self {
        Self { handle, _phantom: PhantomData }
    }
}

impl TryFrom<HandleOrNull> for OwnedHandle {
    type Error = ();

    #[inline]
    fn try_from(handle_or_null: HandleOrNull) -> Result<Self, ()> {
        let owned_handle = handle_or_null.0;
        if owned_handle.handle.is_null() { Err(()) } else { Ok(owned_handle) }
    }
}

impl OwnedHandle {
    /// Creates a new `OwnedHandle` instance that shares the same underlying file handle
    /// as the existing `OwnedHandle` instance.
    pub fn try_clone(&self) -> crate::io::Result<Self> {
        self.duplicate(0, false, c::DUPLICATE_SAME_ACCESS)
    }

    pub(crate) fn duplicate(
        &self,
        access: c::DWORD,
        inherit: bool,
        options: c::DWORD,
    ) -> io::Result<Self> {
        let mut ret = 0 as c::HANDLE;
        cvt(unsafe {
            let cur_proc = c::GetCurrentProcess();
            c::DuplicateHandle(
                cur_proc,
                self.as_raw_handle(),
                cur_proc,
                &mut ret,
                access,
                inherit as c::BOOL,
                options,
            )
        })?;
        unsafe { Ok(Self::from_raw_handle(ret)) }
    }
}

impl TryFrom<HandleOrInvalid> for OwnedHandle {
    type Error = ();

    #[inline]
    fn try_from(handle_or_invalid: HandleOrInvalid) -> Result<Self, ()> {
        let owned_handle = handle_or_invalid.0;
        if owned_handle.handle == c::INVALID_HANDLE_VALUE { Err(()) } else { Ok(owned_handle) }
    }
}

impl AsRawHandle for BorrowedHandle<'_> {
    #[inline]
    fn as_raw_handle(&self) -> RawHandle {
        self.handle
    }
}

impl AsRawHandle for OwnedHandle {
    #[inline]
    fn as_raw_handle(&self) -> RawHandle {
        self.handle
    }
}

impl IntoRawHandle for OwnedHandle {
    #[inline]
    fn into_raw_handle(self) -> RawHandle {
        let handle = self.handle;
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
    /// Note that it *may* have the value `INVALID_HANDLE_VALUE` (-1), which is
    /// sometimes a valid handle value. See [here] for the full story.
    ///
    /// [`RegCloseKey`]: https://docs.microsoft.com/en-us/windows/win32/api/winreg/nf-winreg-regclosekey
    /// [here]: https://devblogs.microsoft.com/oldnewthing/20040302-00/?p=40443
    #[inline]
    unsafe fn from_raw_handle(handle: RawHandle) -> Self {
        Self { handle }
    }
}

impl FromRawHandle for HandleOrNull {
    /// Constructs a new instance of `Self` from the given `RawHandle` returned
    /// from a Windows API that uses null to indicate failure, such as
    /// `CreateThread`.
    ///
    /// Use `HandleOrInvalid` instead of `HandleOrNull` for APIs that
    /// use `INVALID_HANDLE_VALUE` to indicate failure.
    ///
    /// # Safety
    ///
    /// The resource pointed to by `handle` must be either open and otherwise
    /// unowned, or null. Note that not all Windows APIs use null for errors;
    /// see [here] for the full story.
    ///
    /// [here]: https://devblogs.microsoft.com/oldnewthing/20040302-00/?p=40443
    #[inline]
    unsafe fn from_raw_handle(handle: RawHandle) -> Self {
        Self(OwnedHandle::from_raw_handle(handle))
    }
}

impl FromRawHandle for HandleOrInvalid {
    /// Constructs a new instance of `Self` from the given `RawHandle` returned
    /// from a Windows API that uses `INVALID_HANDLE_VALUE` to indicate
    /// failure, such as `CreateFileW`.
    ///
    /// Use `HandleOrNull` instead of `HandleOrInvalid` for APIs that
    /// use null to indicate failure.
    ///
    /// # Safety
    ///
    /// The resource pointed to by `handle` must be either open and otherwise
    /// unowned, null, or equal to `INVALID_HANDLE_VALUE` (-1). Note that not
    /// all Windows APIs use `INVALID_HANDLE_VALUE` for errors; see [here] for
    /// the full story.
    ///
    /// [here]: https://devblogs.microsoft.com/oldnewthing/20040302-00/?p=40443
    #[inline]
    unsafe fn from_raw_handle(handle: RawHandle) -> Self {
        Self(OwnedHandle::from_raw_handle(handle))
    }
}

impl Drop for OwnedHandle {
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

/// A trait to borrow the handle from an underlying object.
#[unstable(feature = "io_safety", issue = "87074")]
pub trait AsHandle {
    /// Borrows the handle.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # #![feature(io_safety)]
    /// use std::fs::File;
    /// # use std::io;
    /// use std::os::windows::io::{AsHandle, BorrowedHandle};
    ///
    /// let mut f = File::open("foo.txt")?;
    /// let borrowed_handle: BorrowedHandle<'_> = f.as_handle();
    /// # Ok::<(), io::Error>(())
    /// ```
    fn as_handle(&self) -> BorrowedHandle<'_>;
}

#[unstable(feature = "io_safety", issue = "87074")]
impl<T: AsHandle> AsHandle for &T {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        T::as_handle(self)
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl<T: AsHandle> AsHandle for &mut T {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        T::as_handle(self)
    }
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
