use std::{borrow::Borrow, fmt, hash, ops::Deref, sync::Arc};

/// A `SmolStr` is a string type that has the following properties:
///
/// * `size_of::<SmolStr>() == size_of::<String>()`
/// * `Clone` is `O(1)`
/// * Strings are stack-allocated if they are:
///     * Up to 22 bytes long
///     * Longer than 22 bytes, but substrings of `WS` (see below). Such strings consist
///     solely of consecutive newlines, followed by consecutive spaces
/// * If a string does not satisfy the aforementioned conditions, it is heap-allocated
///
/// Unlike `String`, however, `SmolStr` is immutable. The primary use case for
/// `SmolStr` is a good enough default storage for tokens of typical programming
/// languages. Strings consisting of a series of newlines, followed by a series of
/// whitespace are a typical pattern in computer programs because of indentation.
/// Note that a specialized interner might be a better solution for some use cases.
#[derive(Clone)]
pub struct SmolStr(Repr);

impl SmolStr {
    pub fn new<T>(text: T) -> SmolStr
    where
        T: Into<String> + AsRef<str>,
    {
        SmolStr(Repr::new(text))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    pub fn to_string(&self) -> String {
        self.as_str().to_string()
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Default for SmolStr {
    fn default() -> SmolStr {
        SmolStr::new("")
    }
}

impl Deref for SmolStr {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl PartialEq<SmolStr> for SmolStr {
    fn eq(&self, other: &SmolStr) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for SmolStr {}

impl PartialEq<str> for SmolStr {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<SmolStr> for str {
    fn eq(&self, other: &SmolStr) -> bool {
        other == self
    }
}

impl<'a> PartialEq<&'a str> for SmolStr {
    fn eq(&self, other: &&'a str) -> bool {
        self == *other
    }
}

impl<'a> PartialEq<SmolStr> for &'a str {
    fn eq(&self, other: &SmolStr) -> bool {
        *self == other
    }
}

impl PartialEq<String> for SmolStr {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<SmolStr> for String {
    fn eq(&self, other: &SmolStr) -> bool {
        other == self
    }
}

impl<'a> PartialEq<&'a String> for SmolStr {
    fn eq(&self, other: &&'a String) -> bool {
        self == *other
    }
}

impl<'a> PartialEq<SmolStr> for &'a String {
    fn eq(&self, other: &SmolStr) -> bool {
        *self == other
    }
}

impl hash::Hash for SmolStr {
    fn hash<H: hash::Hasher>(&self, hasher: &mut H) {
        self.as_str().hash(hasher)
    }
}

impl fmt::Debug for SmolStr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl fmt::Display for SmolStr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}

impl<T> From<T> for SmolStr
where
    T: Into<String> + AsRef<str>,
{
    fn from(text: T) -> Self {
        Self::new(text)
    }
}

impl From<SmolStr> for String {
    fn from(text: SmolStr) -> Self {
        text.to_string()
    }
}

impl Borrow<str> for SmolStr {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

const INLINE_CAP: usize = 22;
const N_NEWLINES: usize = 32;
const N_SPACES: usize = 128;
const WS: &str =
    "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n                                                                                                                                ";

#[derive(Clone, Debug)]
enum Repr {
    Heap(Arc<str>),
    Inline { len: u8, buf: [u8; INLINE_CAP] },
    Substring { newlines: usize, spaces: usize },
}

impl Repr {
    fn new<T>(text: T) -> Self
    where
        T: Into<String> + AsRef<str>,
    {
        {
            let text = text.as_ref();

            let len = text.len();
            if len <= INLINE_CAP {
                let mut buf = [0; INLINE_CAP];
                buf[..len].copy_from_slice(text.as_bytes());
                return Repr::Inline {
                    len: len as u8,
                    buf,
                };
            }

            let newlines = text.bytes().take_while(|&b| b == b'\n').count();
            let spaces = text[newlines..].bytes().take_while(|&b| b == b' ').count();
            if newlines + spaces == len && newlines <= N_NEWLINES && spaces <= N_SPACES {
                return Repr::Substring { newlines, spaces };
            }
        }

        Repr::Heap(text.into().into_boxed_str().into())
    }

    #[inline(always)]
    fn len(&self) -> usize {
        match self {
            Repr::Heap(data) => data.len(),
            Repr::Inline { len, .. } => *len as usize,
            Repr::Substring { newlines, spaces } => *newlines + *spaces,
        }
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        match self {
            Repr::Heap(data) => data.is_empty(),
            Repr::Inline { len, .. } => *len == 0,
            // A substring isn't created for an empty string.
            Repr::Substring { .. } => false,
        }
    }

    fn as_str(&self) -> &str {
        match self {
            Repr::Heap(data) => &*data,
            Repr::Inline { len, buf } => {
                let len = *len as usize;
                let buf = &buf[..len];
                unsafe { ::std::str::from_utf8_unchecked(buf) }
            }
            Repr::Substring { newlines, spaces } => {
                let newlines = *newlines;
                let spaces = *spaces;
                assert!(newlines <= N_NEWLINES && spaces <= N_SPACES);
                &WS[N_NEWLINES - newlines..N_NEWLINES + spaces]
            }
        }
    }
}

#[cfg(feature = "serde")]
mod serde {
    extern crate serde;

    use SmolStr;

    impl serde::Serialize for SmolStr {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            self.as_str().serialize(serializer)
        }
    }

    impl<'de> serde::Deserialize<'de> for SmolStr {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            <&'de str>::deserialize(deserializer).map(SmolStr::from)
        }
    }
}
