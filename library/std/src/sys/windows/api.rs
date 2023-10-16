//! # Safe(r) wrappers around Windows API functions.
//!
//! This module contains fairly thin wrappers around Windows API functions,
//! aimed at centralising safety instead of having unsafe blocks spread
//! throughout higher level code. This makes it much easier to audit FFI safety.
//!
//! Not all functions can be made completely safe without more context but in
//! such cases we should still endeavour to reduce the caller's burden of safety
//! as much as possible.
//!
//! ## Guidelines for wrappers
//!
//! Items here should be named similarly to their raw Windows API name, except
//! that they follow Rust's case conventions. E.g. function names are
//! lower_snake_case. The idea here is that it should be easy for a Windows
//! C/C++ programmer to identify the underlying function that's being wrapped
//! while not looking too out of place in Rust code.
//!
//! Every use of an `unsafe` block must have a related SAFETY comment, even if
//! it's trivially safe (for example, see `get_last_error`). Public unsafe
//! functions must document what the caller has to do to call them safely.
//!
//! Avoid unchecked `as` casts. For integers, either assert that the integer
//! is in range or use `try_into` instead. For pointers, prefer to use
//! `ptr.cast::<Type>()` when possible.
//!
//! This module must only depend on core and not on std types as the eventual
//! hope is to have std depend on sys and not the other way around.
//! However, some amount of glue code may currently be necessary so such code
//! should go in sys/windows/mod.rs rather than here. See `IoResult` as an example.

use core::ffi::c_void;
use core::ptr::addr_of;

use super::c;

/// Helper method for getting the size of `T` as a u32.
/// Errors at compile time if the size would overflow.
///
/// While a type larger than u32::MAX is unlikely, it is possible if only because of a bug.
/// However, one key motivation for this function is to avoid the temptation to
/// use frequent `as` casts. This is risky because they are too powerful.
/// For example, the following will compile today:
///
/// `std::mem::size_of::<u64> as u32`
///
/// Note that `size_of` is never actually called, instead a function pointer is
/// converted to a `u32`. Clippy would warn about this but, alas, it's not run
/// on the standard library.
const fn win32_size_of<T: Sized>() -> u32 {
    // Const assert that the size is less than u32::MAX.
    // Uses a trait to workaround restriction on using generic types in inner items.
    trait Win32SizeOf: Sized {
        const WIN32_SIZE_OF: u32 = {
            let size = core::mem::size_of::<Self>();
            assert!(size <= u32::MAX as usize);
            size as u32
        };
    }
    impl<T: Sized> Win32SizeOf for T {}

    T::WIN32_SIZE_OF
}

/// The `SetFileInformationByHandle` function takes a generic parameter by
/// making the user specify the type (class), a pointer to the data and its
/// size. This trait allows attaching that information to a Rust type so that
/// [`set_file_information_by_handle`] can be called safely.
///
/// This trait is designed so that it can support variable sized types.
/// However, currently Rust's std only uses fixed sized structures.
///
/// # Safety
///
/// * `as_ptr` must return a pointer to memory that is readable up to `size` bytes.
/// * `CLASS` must accurately reflect the type pointed to by `as_ptr`. E.g.
/// the `FILE_BASIC_INFO` structure has the class `FileBasicInfo`.
pub unsafe trait SetFileInformation {
    /// The type of information to set.
    const CLASS: i32;
    /// A pointer to the file information to set.
    fn as_ptr(&self) -> *const c_void;
    /// The size of the type pointed to by `as_ptr`.
    fn size(&self) -> u32;
}
/// Helper trait for implementing `SetFileInformation` for statically sized types.
unsafe trait SizedSetFileInformation: Sized {
    const CLASS: i32;
}
unsafe impl<T: SizedSetFileInformation> SetFileInformation for T {
    const CLASS: i32 = T::CLASS;
    fn as_ptr(&self) -> *const c_void {
        addr_of!(*self).cast::<c_void>()
    }
    fn size(&self) -> u32 {
        win32_size_of::<Self>()
    }
}

// SAFETY: FILE_BASIC_INFO, FILE_END_OF_FILE_INFO, FILE_ALLOCATION_INFO,
// FILE_DISPOSITION_INFO, FILE_DISPOSITION_INFO_EX and FILE_IO_PRIORITY_HINT_INFO
// are all plain `repr(C)` structs that only contain primitive types.
// The given information classes correctly match with the struct.
unsafe impl SizedSetFileInformation for c::FILE_BASIC_INFO {
    const CLASS: i32 = c::FileBasicInfo;
}
unsafe impl SizedSetFileInformation for c::FILE_END_OF_FILE_INFO {
    const CLASS: i32 = c::FileEndOfFileInfo;
}
unsafe impl SizedSetFileInformation for c::FILE_ALLOCATION_INFO {
    const CLASS: i32 = c::FileAllocationInfo;
}
unsafe impl SizedSetFileInformation for c::FILE_DISPOSITION_INFO {
    const CLASS: i32 = c::FileDispositionInfo;
}
unsafe impl SizedSetFileInformation for c::FILE_DISPOSITION_INFO_EX {
    const CLASS: i32 = c::FileDispositionInfoEx;
}
unsafe impl SizedSetFileInformation for c::FILE_IO_PRIORITY_HINT_INFO {
    const CLASS: i32 = c::FileIoPriorityHintInfo;
}

#[inline]
pub fn set_file_information_by_handle<T: SetFileInformation>(
    handle: c::HANDLE,
    info: &T,
) -> Result<(), WinError> {
    unsafe fn set_info(
        handle: c::HANDLE,
        class: i32,
        info: *const c_void,
        size: u32,
    ) -> Result<(), WinError> {
        let result = c::SetFileInformationByHandle(handle, class, info, size);
        (result != 0).then_some(()).ok_or_else(|| get_last_error())
    }
    // SAFETY: The `SetFileInformation` trait ensures that this is safe.
    unsafe { set_info(handle, T::CLASS, info.as_ptr(), info.size()) }
}

/// Gets the error from the last function.
/// This must be called immediately after the function that sets the error to
/// avoid the risk of another function overwriting it.
pub fn get_last_error() -> WinError {
    // SAFETY: This just returns a thread-local u32 and has no other effects.
    unsafe { WinError { code: c::GetLastError() } }
}

/// An error code as returned by [`get_last_error`].
///
/// This is usually a 16-bit Win32 error code but may be a 32-bit HRESULT or NTSTATUS.
/// Check the documentation of the Windows API function being called for expected errors.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct WinError {
    pub code: u32,
}
