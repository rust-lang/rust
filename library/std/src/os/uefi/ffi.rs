use crate::ffi::{OsStr, OsString};
use crate::sealed::Sealed;
use crate::sys_common::ucs2;

#[unstable(feature = "uefi_std", issue = "100499")]
pub trait OsStrExt: Sealed {
    /// This function does not do any allocation
    fn to_ucs2<'a>(&'a self) -> ucs2::EncodeUcs2<'a>;

    /// Creates a UCS-2 Vec which can be passed for FFI.
    /// Note: This function will replace `NULL` and other characters which are not valid in UEFI
    /// strings with `std::sys_common::ucs::Ucs2Char::REPLACEMENT_CHARACTER`
    fn to_ffi_string(&self) -> Vec<u16> {
        let mut v: Vec<u16> = self
            .to_ucs2()
            .map(|x| match x {
                Ok(c) => c,
                Err(_) => ucs2::Ucs2Char::REPLACEMENT_CHARACTER,
            })
            .map(u16::from)
            .collect();
        v.push(0);
        v.shrink_to_fit();
        v
    }
}

impl OsStrExt for OsStr {
    fn to_ucs2<'a>(&'a self) -> ucs2::EncodeUcs2<'a> {
        // This conversion should never fail since the underlying OsStr is UTF-8 encoded
        ucs2::EncodeUcs2::from_str(self.to_str().unwrap())
    }
}

#[unstable(feature = "uefi_std", issue = "100499")]
pub trait OsStringExt: Sealed
where
    Self: Sized,
{
    fn from_ucs2_lossy(ucs: &[u16]) -> Self;

    // For Null terminated UCS-2 Strings.
    #[inline]
    fn from_ucs2_null_termintated(ucs: &[u16]) -> Self {
        Self::from_ucs2_lossy(&ucs[..(ucs.len() - 1)])
    }

    // Create OsString from an FFI obtained pointer.
    // Len is the number of elemented in the string, not number of bytes.
    // Note: This string is assumed to be null terminated
    #[inline]
    unsafe fn from_ffi(ucs: *mut u16, len: usize) -> Self {
        let s = crate::slice::from_raw_parts(ucs, len);
        Self::from_ucs2_null_termintated(s)
    }
}

impl OsStringExt for OsString {
    fn from_ucs2_lossy(ucs: &[u16]) -> Self {
        // Min capacity(in case of all ASCII) is `ucs.len()`
        let mut buf = String::with_capacity(ucs.len());

        for i in ucs {
            let c = match ucs2::Ucs2Char::from_u16(*i) {
                None => char::REPLACEMENT_CHARACTER,
                Some(x) => char::from(x),
            };
            buf.push(c);
        }

        Self::from(buf)
    }
}
