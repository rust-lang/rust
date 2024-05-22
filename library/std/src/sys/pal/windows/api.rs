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
//! should go in sys/pal/windows/mod.rs rather than here. See `IoResult` as an example.

use core::ffi::c_void;
use core::ptr::addr_of;

use super::c;

/// Creates a null-terminated UTF-16 string from a str.
pub macro wide_str($str:literal) {{
    const _: () = {
        if core::slice::memchr::memchr(0, $str.as_bytes()).is_some() {
            panic!("null terminated strings cannot contain interior nulls");
        }
    };
    crate::sys::pal::windows::api::utf16!(concat!($str, '\0'))
}}

/// Creates a UTF-16 string from a str without null termination.
pub macro utf16($str:expr) {{
    const UTF8: &str = $str;
    const UTF16_LEN: usize = crate::sys::pal::windows::api::utf16_len(UTF8);
    const UTF16: [u16; UTF16_LEN] = crate::sys::pal::windows::api::to_utf16(UTF8);
    &UTF16
}}

#[cfg(test)]
mod tests;

/// Gets the UTF-16 length of a UTF-8 string, for use in the wide_str macro.
pub const fn utf16_len(s: &str) -> usize {
    let s = s.as_bytes();
    let mut i = 0;
    let mut len = 0;
    while i < s.len() {
        // the length of a UTF-8 encoded code-point is given by the number of
        // leading ones, except in the case of ASCII.
        let utf8_len = match s[i].leading_ones() {
            0 => 1,
            n => n as usize,
        };
        i += utf8_len;
        // Note that UTF-16 surrogates (U+D800 to U+DFFF) are not encodable as UTF-8,
        // so (unlike with WTF-8) we don't have to worry about how they'll get re-encoded.
        len += if utf8_len < 4 { 1 } else { 2 };
    }
    len
}

/// Const convert UTF-8 to UTF-16, for use in the wide_str macro.
///
/// Note that this is designed for use in const contexts so is not optimized.
pub const fn to_utf16<const UTF16_LEN: usize>(s: &str) -> [u16; UTF16_LEN] {
    let mut output = [0_u16; UTF16_LEN];
    let mut pos = 0;
    let s = s.as_bytes();
    let mut i = 0;
    while i < s.len() {
        match s[i].leading_ones() {
            // Decode UTF-8 based on its length.
            // See https://en.wikipedia.org/wiki/UTF-8
            0 => {
                // ASCII is the same in both encodings
                output[pos] = s[i] as u16;
                i += 1;
                pos += 1;
            }
            2 => {
                // Bits: 110xxxxx 10xxxxxx
                output[pos] = ((s[i] as u16 & 0b11111) << 6) | (s[i + 1] as u16 & 0b111111);
                i += 2;
                pos += 1;
            }
            3 => {
                // Bits: 1110xxxx 10xxxxxx 10xxxxxx
                output[pos] = ((s[i] as u16 & 0b1111) << 12)
                    | ((s[i + 1] as u16 & 0b111111) << 6)
                    | (s[i + 2] as u16 & 0b111111);
                i += 3;
                pos += 1;
            }
            4 => {
                // Bits: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
                let mut c = ((s[i] as u32 & 0b111) << 18)
                    | ((s[i + 1] as u32 & 0b111111) << 12)
                    | ((s[i + 2] as u32 & 0b111111) << 6)
                    | (s[i + 3] as u32 & 0b111111);
                // re-encode as UTF-16 (see https://en.wikipedia.org/wiki/UTF-16)
                // - Subtract 0x10000 from the code point
                // - For the high surrogate, shift right by 10 then add 0xD800
                // - For the low surrogate, take the low 10 bits then add 0xDC00
                c -= 0x10000;
                output[pos] = ((c >> 10) + 0xD800) as u16;
                output[pos + 1] = ((c & 0b1111111111) + 0xDC00) as u16;
                i += 4;
                pos += 2;
            }
            // valid UTF-8 cannot have any other values
            _ => unreachable!(),
        }
    }
    output
}

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
    // Const assert that the size does not exceed u32::MAX.
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
        (result != 0).then_some(()).ok_or_else(get_last_error)
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
