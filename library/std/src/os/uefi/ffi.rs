use crate::ffi::{OsStr, OsString};
use crate::sealed::Sealed;
use crate::sys_common::ucs2;

const REPLACEMENT_CHARACTER_UCS2: u16 = 0xfffdu16;

#[unstable(feature = "uefi_std", issue = "none")]
pub trait OsStrExt: Sealed {
    fn to_ucs2<'a>(&'a self) -> ucs2::EncodeUcs2<'a>;
}

impl OsStrExt for OsStr {
    fn to_ucs2<'a>(&'a self) -> ucs2::EncodeUcs2<'a> {
        // This conversion should never fail since the underlying OsStr is UTF-8 encoded
        ucs2::EncodeUcs2::from_str(self.to_str().unwrap())
    }
}

#[unstable(feature = "uefi_std", issue = "none")]
pub trait OsStringExt: Sealed
where
    Self: Sized,
{
    fn from_ucs2(ucs: &[u16]) -> Self;

    // For Null terminated UCS-2 Strings
    fn from_ucs2_null_termintated(ucs: &[u16]) -> Self {
        Self::from_ucs2(&ucs[..(ucs.len() - 1)])
    }
}

impl OsStringExt for OsString {
    fn from_ucs2(ucs: &[u16]) -> Self {
        // Min capacity(in case of all ASCII) is `ucs.len()`
        let mut buf = String::with_capacity(ucs.len());

        for i in ucs {
            let c = char::from(ucs2::Ucs2Char::from_u16(*i));
            buf.push(c);
        }

        Self::from(buf)
    }
}
