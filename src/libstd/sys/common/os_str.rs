/// The underlying OsString/OsStr implementation on Unix systems: just
/// a `Vec<u8>`/`[u8]`.
#[cfg(not(target_family = "windows"))]
pub mod u8 {
    use string::String;
    use vec::Vec;
    use borrow;
    use str;
    use fmt;
    use mem;
    use ops;

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
            self.to_string_lossy().fmt(formatter)
        }
    }

    impl fmt::Debug for OsString {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
            borrow::Borrow::<OsStr>::borrow(self).fmt(formatter)
        }
    }

    impl OsString {
        pub fn from_string(s: String) -> Self {
            OsString { inner: s.into_bytes() }
        }

        pub fn from_bytes(b: Vec<u8>) -> Option<Self> {
            Some(Self::from_vec(b))
        }

        pub fn into_string(self) -> Result<String, Self> {
            String::from_utf8(self.inner).map_err(|p| OsString { inner: p.into_bytes() } )
        }

        pub fn as_mut_vec(&mut self) -> &mut Vec<u8> {
            &mut self.inner
        }

        pub fn push_slice(&mut self, s: &OsStr) {
            self.inner.push_all(&s.inner)
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

    impl OsStr {
        pub fn from_str(s: &str) -> &OsStr {
            Self::from_bytes(s.as_bytes())
        }

        pub fn to_bytes(&self) -> Option<&[u8]> {
            Some(self.as_bytes())
        }

        pub fn as_bytes(&self) -> &[u8] {
            &self.inner
        }

        pub fn to_str(&self) -> Option<&str> {
            str::from_utf8(&self.inner).ok()
        }

        pub fn to_string_lossy(&self) -> borrow::Cow<str> {
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

#[cfg(target_family = "windows")]
pub mod wtf8 {
    use string::String;
    use vec::Vec;
    use sys::wtf8::{Wtf8, Wtf8Buf};
    use borrow;
    use fmt;
    use mem;
    use ops;

    #[derive(Clone, Hash)]
    pub struct OsString {
        pub inner: Wtf8Buf
    }

    pub struct OsStr {
        pub inner: Wtf8
    }

    impl fmt::Debug for OsStr {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
            self.to_string_lossy().fmt(formatter)
        }
    }

    impl fmt::Debug for OsString {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
            borrow::Borrow::<OsStr>::borrow(self).fmt(formatter)
        }
    }

    impl OsString {
        pub fn from_string(s: String) -> Self {
            OsString { inner: Wtf8Buf::from_string(s) }
        }

        pub fn from_bytes(b: Vec<u8>) -> Option<Self> {
            String::from_utf8(b).ok().map(Self::from_string)
        }

        pub fn into_string(self) -> Result<String, Self> {
            self.inner.into_string().map_err(|buf| OsString { inner: buf })
        }

        pub fn as_mut_vec(&mut self) -> &mut Vec<u8> {
            unsafe { mem::transmute(&mut self.inner) }
        }

        pub fn push_slice(&mut self, s: &OsStr) {
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

    impl OsStr {
        pub fn from_str(s: &str) -> &Self {
            unsafe { mem::transmute(Wtf8::from_str(s)) }
        }

        pub fn to_bytes(&self) -> Option<&[u8]> {
            self.to_str().map(|s| s.as_bytes())
        }

        pub fn as_bytes(&self) -> &[u8] {
            use sys::inner::*;

            self.inner.as_inner()
        }

        pub fn to_str(&self) -> Option<&str> {
            self.inner.as_str()
        }

        pub fn to_string_lossy(&self) -> borrow::Cow<str> {
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
