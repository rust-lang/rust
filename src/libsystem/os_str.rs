pub use imp::os_str as imp;

pub mod traits {
    pub use os_str::{OsStr as sys_OsStr, OsString as sys_OsString};
}

pub mod prelude {
    pub use super::imp::{OsStr, OsString};
    pub use super::traits::*;
}

use collections::{String, Vec};
use collections::borrow;
use core::hash;
use core::fmt;
use core::ops;

pub trait OsStr: borrow::ToOwned + fmt::Debug {
    type OsString: OsString<Self>;

    fn from_str(s: &str) -> &Self;

    fn to_str(&self) -> Option<&str>;
    fn to_string_lossy(&self) -> borrow::Cow<str>;

    fn to_bytes(&self) -> Option<&[u8]>;
    fn as_bytes(&self) -> &[u8];
}

pub trait OsString<S: OsStr + ?Sized>: borrow::Borrow<S> + ops::Deref<Target=S> + Clone + hash::Hash + fmt::Debug + Sized {
    fn from_string(s: String) -> Self;
    fn from_bytes(b: Vec<u8>) -> Option<Self>;

    fn into_string(self) -> Result<String, Self>;
    fn push_slice(&mut self, s: &S);

    fn as_mut_vec(&mut self) -> &mut Vec<u8>;
}

/// The underlying OsString/OsStr implementation on Unix systems: just
/// a `Vec<u8>`/`[u8]`.
pub mod u8 {
    use collections::{String, Vec};
    use collections::borrow;
    use core::str;
    use core::fmt;
    use core::mem;
    use core::ops;

    #[derive(Clone, PartialOrd, Ord, PartialEq, Eq, Hash)]
    pub struct OsString {
        pub inner: Vec<u8>
    }

    #[derive(PartialOrd, Ord, PartialEq, Eq, Hash)]
    pub struct OsStr {
        pub inner: [u8]
    }

    impl fmt::Debug for OsStr {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
            super::OsStr::to_string_lossy(self).fmt(formatter)
        }
    }

    impl fmt::Debug for OsString {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
            borrow::Borrow::<OsStr>::borrow(self).fmt(formatter)
        }
    }

    impl super::OsString<OsStr> for OsString {
        fn from_string(s: String) -> Self {
            OsString { inner: s.into_bytes() }
        }

        fn from_bytes(b: Vec<u8>) -> Option<Self> {
            Some(Self::from_vec(b))
        }

        fn into_string(self) -> Result<String, Self> {
            String::from_utf8(self.inner).map_err(|p| OsString { inner: p.into_bytes() } )
        }

        fn as_mut_vec(&mut self) -> &mut Vec<u8> {
            &mut self.inner
        }

        fn push_slice(&mut self, s: &OsStr) {
            self.inner.push_all(&s.inner)
        }
    }

    impl OsString {
        pub fn path_join(&mut self, path: &OsStr) {
            use super::prelude::*;
            use path::prelude::*;

            let need_sep = self.as_bytes().last().map(|c| !PathInfo::is_sep_byte(*c)).unwrap_or(false);
            let is_absolute = self.as_bytes().first().map(|c| PathInfo::is_sep_byte(*c)).unwrap_or(false);

            if is_absolute {
                self.as_mut_vec().truncate(0);
            } else if need_sep {
                self.push_slice(OsStr::from_str(PathInfo::MAIN_SEP_STR));
            }

            self.push_slice(path);
        }
    }

    impl OsString {
        pub fn from_vec(b: Vec<u8>) -> Self {
            OsString { inner: b }
        }
    }

    impl borrow::Borrow<OsStr> for OsString {
        fn borrow(&self) -> &OsStr {
            ops::Deref::deref(self)
        }
    }

    impl ops::Deref for OsString {
        type Target = OsStr;

        fn deref(&self) -> &OsStr {
            unsafe { mem::transmute(&*self.inner) }
        }
    }

    impl super::OsStr for OsStr {
        type OsString = OsString;

        fn from_str(s: &str) -> &OsStr {
            Self::from_bytes(s.as_bytes())
        }

        fn to_bytes(&self) -> Option<&[u8]> {
            Some(self.as_bytes())
        }

        fn as_bytes(&self) -> &[u8] {
            &self.inner
        }

        fn to_str(&self) -> Option<&str> {
            str::from_utf8(&self.inner).ok()
        }

        fn to_string_lossy(&self) -> borrow::Cow<str> {
            String::from_utf8_lossy(&self.inner)
        }
    }

    impl OsStr {
        pub fn from_bytes(b: &[u8]) -> &OsStr {
            unsafe { mem::transmute(b) }
        }
    }

    impl borrow::ToOwned for OsStr {
        type Owned = OsString;

        fn to_owned(&self) -> OsString {
            OsString { inner: self.inner.to_vec() }
        }
    }
}

pub mod wtf8 {
    use collections::{String, Vec};
    use wtf8::{Wtf8, Wtf8Buf};
    use collections::borrow;
    use core::fmt;
    use core::mem;
    use core::ops;

    #[derive(Clone, Hash)]
    pub struct OsString {
        pub inner: Wtf8Buf
    }

    pub struct OsStr {
        pub inner: Wtf8
    }

    impl fmt::Debug for OsStr {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
            super::OsStr::to_string_lossy(self).fmt(formatter)
        }
    }

    impl fmt::Debug for OsString {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
            borrow::Borrow::<OsStr>::borrow(self).fmt(formatter)
        }
    }

    impl super::OsString<OsStr> for OsString {
        fn from_string(s: String) -> Self {
            OsString { inner: Wtf8Buf::from_string(s) }
        }

        fn from_bytes(b: Vec<u8>) -> Option<Self> {
            String::from_utf8(b).ok().map(Self::from_string)
        }

        fn into_string(self) -> Result<String, Self> {
            self.inner.into_string().map_err(|buf| OsString { inner: buf })
        }

        fn as_mut_vec(&mut self) -> &mut Vec<u8> {
            unsafe { mem::transmute(&mut self.inner) }
        }

        fn push_slice(&mut self, s: &OsStr) {
            self.inner.push_wtf8(&s.inner)
        }
    }

    impl borrow::Borrow<OsStr> for OsString {
        fn borrow(&self) -> &OsStr {
            ops::Deref::deref(self)
        }
    }

    impl ops::Deref for OsString {
        type Target = OsStr;

        fn deref(&self) -> &OsStr {
            unsafe { mem::transmute(self.inner.as_slice()) }
        }
    }

    impl super::OsStr for OsStr {
        type OsString = OsString;

        fn from_str(s: &str) -> &Self {
            unsafe { mem::transmute(Wtf8::from_str(s)) }
        }

        fn to_bytes(&self) -> Option<&[u8]> {
            self.to_str().map(|s| s.as_bytes())
        }

        fn as_bytes(&self) -> &[u8] {
            use inner::prelude::*;

            self.inner.as_inner()
        }

        fn to_str(&self) -> Option<&str> {
            self.inner.as_str()
        }

        fn to_string_lossy(&self) -> borrow::Cow<str> {
            self.inner.to_string_lossy()
        }
    }

    impl borrow::ToOwned for OsStr {
        type Owned = OsString;

        fn to_owned(&self) -> OsString {
            let mut buf = Wtf8Buf::with_capacity(self.inner.len());
            buf.push_wtf8(&self.inner);
            OsString { inner: buf }
        }
    }
}
