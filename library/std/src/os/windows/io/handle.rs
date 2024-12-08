//! Owned and borrowed OS handles.

#![stable(feature = "io_safety", since = "1.63.0")]

use super::raw::{AsRawHandle, FromRawHandle, IntoRawHandle, RawHandle};
use crate::marker::PhantomData;
use crate::mem::ManuallyDrop;
use crate::sys::cvt;
use crate::sys_common::{AsInner, FromInner, IntoInner};
use crate::{fmt, fs, io, ptr, sys};

/// A borrowed handle.
///
/// This has a lifetime parameter to tie it to the lifetime of something that
/// owns the handle.
///
/// This uses `repr(transparent)` and has the representation of a host handle,
/// so it can be used in FFI in places where a handle is passed as an argument,
/// it is not captured or consumed.
///
/// Note that it *may* have the value `-1`, which in `BorrowedHandle` always
/// represents a valid handle value, such as [the current process handle], and
/// not `INVALID_HANDLE_VALUE`, despite the two having the same value. See
/// [here] for the full story.
///
/// And, it *may* have the value `NULL` (0), which can occur when consoles are
/// detached from processes, or when `windows_subsystem` is used.
///
/// This type's `.to_owned()` implementation returns another `BorrowedHandle`
/// rather than an `OwnedHandle`. It just makes a trivial copy of the raw
/// handle, which is then borrowed under the same lifetime.
///
/// [here]: https://devblogs.microsoft.com/oldnewthing/20040302-00/?p=40443
/// [the current process handle]: https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getcurrentprocess#remarks
#[derive(Copy, Clone)]
#[repr(transparent)]
#[stable(feature = "io_safety", since = "1.63.0")]
pub struct BorrowedHandle<'handle> {
    handle: RawHandle,
    _phantom: PhantomData<&'handle OwnedHandle>,
}

/// An owned handle.
///
/// This closes the handle on drop.
///
/// Note that it *may* have the value `-1`, which in `OwnedHandle` always
/// represents a valid handle value, such as [the current process handle], and
/// not `INVALID_HANDLE_VALUE`, despite the two having the same value. See
/// [here] for the full story.
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
/// [the current process handle]: https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getcurrentprocess#remarks
#[repr(transparent)]
#[stable(feature = "io_safety", since = "1.63.0")]
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
/// This type may hold any handle value that [`OwnedHandle`] may hold. As with `OwnedHandle`, when
/// it holds `-1`, that value is interpreted as a valid handle value, such as
/// [the current process handle], and not `INVALID_HANDLE_VALUE`.
///
/// If this holds a non-null handle, it will close the handle on drop.
///
/// [the current process handle]: https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getcurrentprocess#remarks
#[repr(transparent)]
#[stable(feature = "io_safety", since = "1.63.0")]
#[derive(Debug)]
pub struct HandleOrNull(RawHandle);

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
/// This type may hold any handle value that [`OwnedHandle`] may hold, except that when it holds
/// `-1`, that value is interpreted to mean `INVALID_HANDLE_VALUE`.
///
/// If holds a handle other than `INVALID_HANDLE_VALUE`, it will close the handle on drop.
#[repr(transparent)]
#[stable(feature = "io_safety", since = "1.63.0")]
#[derive(Debug)]
pub struct HandleOrInvalid(RawHandle);

// The Windows [`HANDLE`] type may be transferred across and shared between
// thread boundaries (despite containing a `*mut void`, which in general isn't
// `Send` or `Sync`).
//
// [`HANDLE`]: std::os::windows::raw::HANDLE
#[stable(feature = "io_safety", since = "1.63.0")]
unsafe impl Send for OwnedHandle {}
#[stable(feature = "io_safety", since = "1.63.0")]
unsafe impl Send for HandleOrNull {}
#[stable(feature = "io_safety", since = "1.63.0")]
unsafe impl Send for HandleOrInvalid {}
#[stable(feature = "io_safety", since = "1.63.0")]
unsafe impl Send for BorrowedHandle<'_> {}
#[stable(feature = "io_safety", since = "1.63.0")]
unsafe impl Sync for OwnedHandle {}
#[stable(feature = "io_safety", since = "1.63.0")]
unsafe impl Sync for HandleOrNull {}
#[stable(feature = "io_safety", since = "1.63.0")]
unsafe impl Sync for HandleOrInvalid {}
#[stable(feature = "io_safety", since = "1.63.0")]
unsafe impl Sync for BorrowedHandle<'_> {}

impl BorrowedHandle<'_> {
    /// Returns a `BorrowedHandle` holding the given raw handle.
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
    #[rustc_const_stable(feature = "io_safety", since = "1.63.0")]
    #[stable(feature = "io_safety", since = "1.63.0")]
    pub const unsafe fn borrow_raw(handle: RawHandle) -> Self {
        Self { handle, _phantom: PhantomData }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl TryFrom<HandleOrNull> for OwnedHandle {
    type Error = NullHandleError;

    #[inline]
    fn try_from(handle_or_null: HandleOrNull) -> Result<Self, NullHandleError> {
        let handle_or_null = ManuallyDrop::new(handle_or_null);
        if handle_or_null.is_valid() {
            // SAFETY: The handle is not null.
            Ok(unsafe { OwnedHandle::from_raw_handle(handle_or_null.0) })
        } else {
            Err(NullHandleError(()))
        }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl Drop for HandleOrNull {
    #[inline]
    fn drop(&mut self) {
        if self.is_valid() {
            unsafe {
                let _ = sys::c::CloseHandle(self.0);
            }
        }
    }
}

impl OwnedHandle {
    /// Creates a new `OwnedHandle` instance that shares the same underlying
    /// object as the existing `OwnedHandle` instance.
    #[stable(feature = "io_safety", since = "1.63.0")]
    pub fn try_clone(&self) -> crate::io::Result<Self> {
        self.as_handle().try_clone_to_owned()
    }
}

impl BorrowedHandle<'_> {
    /// Creates a new `OwnedHandle` instance that shares the same underlying
    /// object as the existing `BorrowedHandle` instance.
    #[stable(feature = "io_safety", since = "1.63.0")]
    pub fn try_clone_to_owned(&self) -> crate::io::Result<OwnedHandle> {
        self.duplicate(0, false, sys::c::DUPLICATE_SAME_ACCESS)
    }

    pub(crate) fn duplicate(
        &self,
        access: u32,
        inherit: bool,
        options: u32,
    ) -> io::Result<OwnedHandle> {
        let handle = self.as_raw_handle();

        // `Stdin`, `Stdout`, and `Stderr` can all hold null handles, such as
        // in a process with a detached console. `DuplicateHandle` would fail
        // if we passed it a null handle, but we can treat null as a valid
        // handle which doesn't do any I/O, and allow it to be duplicated.
        if handle.is_null() {
            return unsafe { Ok(OwnedHandle::from_raw_handle(handle)) };
        }

        let mut ret = ptr::null_mut();
        cvt(unsafe {
            let cur_proc = sys::c::GetCurrentProcess();
            sys::c::DuplicateHandle(
                cur_proc,
                handle,
                cur_proc,
                &mut ret,
                access,
                inherit as sys::c::BOOL,
                options,
            )
        })?;
        unsafe { Ok(OwnedHandle::from_raw_handle(ret)) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl TryFrom<HandleOrInvalid> for OwnedHandle {
    type Error = InvalidHandleError;

    #[inline]
    fn try_from(handle_or_invalid: HandleOrInvalid) -> Result<Self, InvalidHandleError> {
        let handle_or_invalid = ManuallyDrop::new(handle_or_invalid);
        if handle_or_invalid.is_valid() {
            // SAFETY: The handle is not invalid.
            Ok(unsafe { OwnedHandle::from_raw_handle(handle_or_invalid.0) })
        } else {
            Err(InvalidHandleError(()))
        }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl Drop for HandleOrInvalid {
    #[inline]
    fn drop(&mut self) {
        if self.is_valid() {
            unsafe {
                let _ = sys::c::CloseHandle(self.0);
            }
        }
    }
}

/// This is the error type used by [`HandleOrNull`] when attempting to convert
/// into a handle, to indicate that the value is null.
// The empty field prevents constructing this, and allows extending it in the future.
#[stable(feature = "io_safety", since = "1.63.0")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NullHandleError(());

#[stable(feature = "io_safety", since = "1.63.0")]
impl fmt::Display for NullHandleError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        "A HandleOrNull could not be converted to a handle because it was null".fmt(fmt)
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl crate::error::Error for NullHandleError {}

/// This is the error type used by [`HandleOrInvalid`] when attempting to
/// convert into a handle, to indicate that the value is
/// `INVALID_HANDLE_VALUE`.
// The empty field prevents constructing this, and allows extending it in the future.
#[stable(feature = "io_safety", since = "1.63.0")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidHandleError(());

#[stable(feature = "io_safety", since = "1.63.0")]
impl fmt::Display for InvalidHandleError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        "A HandleOrInvalid could not be converted to a handle because it was INVALID_HANDLE_VALUE"
            .fmt(fmt)
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl crate::error::Error for InvalidHandleError {}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsRawHandle for BorrowedHandle<'_> {
    #[inline]
    fn as_raw_handle(&self) -> RawHandle {
        self.handle
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsRawHandle for OwnedHandle {
    #[inline]
    fn as_raw_handle(&self) -> RawHandle {
        self.handle
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl IntoRawHandle for OwnedHandle {
    #[inline]
    fn into_raw_handle(self) -> RawHandle {
        ManuallyDrop::new(self).handle
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl FromRawHandle for OwnedHandle {
    #[inline]
    unsafe fn from_raw_handle(handle: RawHandle) -> Self {
        Self { handle }
    }
}

impl HandleOrNull {
    /// Constructs a new instance of `Self` from the given `RawHandle` returned
    /// from a Windows API that uses null to indicate failure, such as
    /// `CreateThread`.
    ///
    /// Use `HandleOrInvalid` instead of `HandleOrNull` for APIs that
    /// use `INVALID_HANDLE_VALUE` to indicate failure.
    ///
    /// # Safety
    ///
    /// The passed `handle` value must either satisfy the safety requirements
    /// of [`FromRawHandle::from_raw_handle`], or be null. Note that not all
    /// Windows APIs use null for errors; see [here] for the full story.
    ///
    /// [here]: https://devblogs.microsoft.com/oldnewthing/20040302-00/?p=40443
    #[stable(feature = "io_safety", since = "1.63.0")]
    #[inline]
    pub unsafe fn from_raw_handle(handle: RawHandle) -> Self {
        Self(handle)
    }

    fn is_valid(&self) -> bool {
        !self.0.is_null()
    }
}

impl HandleOrInvalid {
    /// Constructs a new instance of `Self` from the given `RawHandle` returned
    /// from a Windows API that uses `INVALID_HANDLE_VALUE` to indicate
    /// failure, such as `CreateFileW`.
    ///
    /// Use `HandleOrNull` instead of `HandleOrInvalid` for APIs that
    /// use null to indicate failure.
    ///
    /// # Safety
    ///
    /// The passed `handle` value must either satisfy the safety requirements
    /// of [`FromRawHandle::from_raw_handle`], or be
    /// `INVALID_HANDLE_VALUE` (-1). Note that not all Windows APIs use
    /// `INVALID_HANDLE_VALUE` for errors; see [here] for the full story.
    ///
    /// [here]: https://devblogs.microsoft.com/oldnewthing/20040302-00/?p=40443
    #[stable(feature = "io_safety", since = "1.63.0")]
    #[inline]
    pub unsafe fn from_raw_handle(handle: RawHandle) -> Self {
        Self(handle)
    }

    fn is_valid(&self) -> bool {
        self.0 != sys::c::INVALID_HANDLE_VALUE
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl Drop for OwnedHandle {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let _ = sys::c::CloseHandle(self.handle);
        }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl fmt::Debug for BorrowedHandle<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BorrowedHandle").field("handle", &self.handle).finish()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl fmt::Debug for OwnedHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnedHandle").field("handle", &self.handle).finish()
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

impl_is_terminal!(BorrowedHandle<'_>, OwnedHandle);

/// A trait to borrow the handle from an underlying object.
#[stable(feature = "io_safety", since = "1.63.0")]
pub trait AsHandle {
    /// Borrows the handle.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::fs::File;
    /// # use std::io;
    /// use std::os::windows::io::{AsHandle, BorrowedHandle};
    ///
    /// let mut f = File::open("foo.txt")?;
    /// let borrowed_handle: BorrowedHandle<'_> = f.as_handle();
    /// # Ok::<(), io::Error>(())
    /// ```
    #[stable(feature = "io_safety", since = "1.63.0")]
    fn as_handle(&self) -> BorrowedHandle<'_>;
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl<T: AsHandle + ?Sized> AsHandle for &T {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        T::as_handle(self)
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl<T: AsHandle + ?Sized> AsHandle for &mut T {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        T::as_handle(self)
    }
}

#[stable(feature = "as_windows_ptrs", since = "1.71.0")]
/// This impl allows implementing traits that require `AsHandle` on Arc.
/// ```
/// # #[cfg(windows)] mod group_cfg {
/// # use std::os::windows::io::AsHandle;
/// use std::fs::File;
/// use std::sync::Arc;
///
/// trait MyTrait: AsHandle {}
/// impl MyTrait for Arc<File> {}
/// impl MyTrait for Box<File> {}
/// # }
/// ```
impl<T: AsHandle + ?Sized> AsHandle for crate::sync::Arc<T> {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        (**self).as_handle()
    }
}

#[stable(feature = "as_windows_ptrs", since = "1.71.0")]
impl<T: AsHandle + ?Sized> AsHandle for crate::rc::Rc<T> {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        (**self).as_handle()
    }
}

#[stable(feature = "as_windows_ptrs", since = "1.71.0")]
impl<T: AsHandle + ?Sized> AsHandle for Box<T> {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        (**self).as_handle()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsHandle for BorrowedHandle<'_> {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        *self
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsHandle for OwnedHandle {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        // Safety: `OwnedHandle` and `BorrowedHandle` have the same validity
        // invariants, and the `BorrowedHandle` is bounded by the lifetime
        // of `&self`.
        unsafe { BorrowedHandle::borrow_raw(self.as_raw_handle()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsHandle for fs::File {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        self.as_inner().as_handle()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<fs::File> for OwnedHandle {
    /// Takes ownership of a [`File`](fs::File)'s underlying file handle.
    #[inline]
    fn from(file: fs::File) -> OwnedHandle {
        file.into_inner().into_inner().into_inner()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<OwnedHandle> for fs::File {
    /// Returns a [`File`](fs::File) that takes ownership of the given handle.
    #[inline]
    fn from(owned: OwnedHandle) -> Self {
        Self::from_inner(FromInner::from_inner(FromInner::from_inner(owned)))
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsHandle for crate::io::Stdin {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        unsafe { BorrowedHandle::borrow_raw(self.as_raw_handle()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl<'a> AsHandle for crate::io::StdinLock<'a> {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        unsafe { BorrowedHandle::borrow_raw(self.as_raw_handle()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsHandle for crate::io::Stdout {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        unsafe { BorrowedHandle::borrow_raw(self.as_raw_handle()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl<'a> AsHandle for crate::io::StdoutLock<'a> {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        unsafe { BorrowedHandle::borrow_raw(self.as_raw_handle()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsHandle for crate::io::Stderr {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        unsafe { BorrowedHandle::borrow_raw(self.as_raw_handle()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl<'a> AsHandle for crate::io::StderrLock<'a> {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        unsafe { BorrowedHandle::borrow_raw(self.as_raw_handle()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsHandle for crate::process::ChildStdin {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        unsafe { BorrowedHandle::borrow_raw(self.as_raw_handle()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<crate::process::ChildStdin> for OwnedHandle {
    /// Takes ownership of a [`ChildStdin`](crate::process::ChildStdin)'s file handle.
    #[inline]
    fn from(child_stdin: crate::process::ChildStdin) -> OwnedHandle {
        unsafe { OwnedHandle::from_raw_handle(child_stdin.into_raw_handle()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsHandle for crate::process::ChildStdout {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        unsafe { BorrowedHandle::borrow_raw(self.as_raw_handle()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<crate::process::ChildStdout> for OwnedHandle {
    /// Takes ownership of a [`ChildStdout`](crate::process::ChildStdout)'s file handle.
    #[inline]
    fn from(child_stdout: crate::process::ChildStdout) -> OwnedHandle {
        unsafe { OwnedHandle::from_raw_handle(child_stdout.into_raw_handle()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsHandle for crate::process::ChildStderr {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        unsafe { BorrowedHandle::borrow_raw(self.as_raw_handle()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<crate::process::ChildStderr> for OwnedHandle {
    /// Takes ownership of a [`ChildStderr`](crate::process::ChildStderr)'s file handle.
    #[inline]
    fn from(child_stderr: crate::process::ChildStderr) -> OwnedHandle {
        unsafe { OwnedHandle::from_raw_handle(child_stderr.into_raw_handle()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl<T> AsHandle for crate::thread::JoinHandle<T> {
    #[inline]
    fn as_handle(&self) -> BorrowedHandle<'_> {
        unsafe { BorrowedHandle::borrow_raw(self.as_raw_handle()) }
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl<T> From<crate::thread::JoinHandle<T>> for OwnedHandle {
    #[inline]
    fn from(join_handle: crate::thread::JoinHandle<T>) -> OwnedHandle {
        join_handle.into_inner().into_handle().into_inner()
    }
}
