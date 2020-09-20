use std::{
    borrow::Borrow,
    cmp::{self, Ordering},
    fmt, hash, iter,
    ops::Deref,
    sync::Arc,
};

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
    #[deprecated = "Use `new_inline` instead"]
    pub const fn new_inline_from_ascii(len: usize, bytes: &[u8]) -> SmolStr {
        let _len_is_short = [(); INLINE_CAP + 1][len];

        const ZEROS: &[u8] = &[0; INLINE_CAP];

        let mut buf = [0; INLINE_CAP];
        macro_rules! s {
            ($($idx:literal),*) => ( $(s!(set $idx);)* );
            (set $idx:literal) => ({
                let src: &[u8] = [ZEROS, bytes][($idx < len) as usize];
                let byte = src[$idx];
                let _is_ascii = [(); 128][byte as usize];
                buf[$idx] = byte
            });
        }
        s!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21);
        SmolStr(Repr::Inline {
            len: len as u8,
            buf,
        })
    }

    /// Constructs inline variant of `SmolStr`.
    ///
    /// Panics if `text.len() > 22`.
    #[inline]
    pub const fn new_inline(text: &str) -> SmolStr {
        let mut buf = [0; INLINE_CAP];
        let mut i = 0;
        while i < text.len() {
            buf[i] = text.as_bytes()[i];
            i += 1
        }
        SmolStr(Repr::Inline {
            len: text.len() as u8,
            buf,
        })
    }

    pub fn new<T>(text: T) -> SmolStr
    where
        T: AsRef<str>,
    {
        SmolStr(Repr::new(text))
    }

    #[inline(always)]
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    #[inline(always)]
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

    #[inline(always)]
    pub fn is_heap_allocated(&self) -> bool {
        match self.0 {
            Repr::Heap(..) => true,
            _ => false,
        }
    }

    fn from_char_iter<I: iter::Iterator<Item = char>>(mut iter: I) -> SmolStr {
        let (min_size, _) = iter.size_hint();
        if min_size > INLINE_CAP {
            let heap: String = iter.collect();
            return SmolStr(Repr::Heap(heap.into_boxed_str().into()));
        }
        let mut len = 0;
        let mut buf = [0u8; INLINE_CAP];
        while let Some(ch) = iter.next() {
            let size = ch.len_utf8();
            if size + len > INLINE_CAP {
                let (min_remaining, _) = iter.size_hint();
                let mut heap = String::with_capacity(size + len + min_remaining);
                heap.push_str(std::str::from_utf8(&buf[..len]).unwrap());
                heap.push(ch);
                heap.extend(iter);
                return SmolStr(Repr::Heap(heap.into_boxed_str().into()));
            }
            ch.encode_utf8(&mut buf[len..]);
            len += size;
        }
        SmolStr(Repr::Inline {
            len: len as u8,
            buf,
        })
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

impl Ord for SmolStr {
    fn cmp(&self, other: &SmolStr) -> Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl PartialOrd for SmolStr {
    fn partial_cmp(&self, other: &SmolStr) -> Option<Ordering> {
        Some(self.cmp(other))
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

impl iter::FromIterator<char> for SmolStr {
    fn from_iter<I: iter::IntoIterator<Item = char>>(iter: I) -> SmolStr {
        let iter = iter.into_iter();
        Self::from_char_iter(iter)
    }
}

fn build_from_str_iter<T>(mut iter: impl Iterator<Item = T>) -> SmolStr
where
    T: AsRef<str>,
    String: iter::Extend<T>,
{
    let mut len = 0;
    let mut buf = [0u8; INLINE_CAP];
    while let Some(slice) = iter.next() {
        let slice = slice.as_ref();
        let size = slice.len();
        if size + len > INLINE_CAP {
            let mut heap = String::with_capacity(size + len);
            heap.push_str(std::str::from_utf8(&buf[..len]).unwrap());
            heap.push_str(&slice);
            heap.extend(iter);
            return SmolStr(Repr::Heap(heap.into_boxed_str().into()));
        }
        (&mut buf[len..][..size]).copy_from_slice(slice.as_bytes());
        len += size;
    }
    SmolStr(Repr::Inline {
        len: len as u8,
        buf,
    })
}

impl iter::FromIterator<String> for SmolStr {
    fn from_iter<I: iter::IntoIterator<Item = String>>(iter: I) -> SmolStr {
        build_from_str_iter(iter.into_iter())
    }
}

impl<'a> iter::FromIterator<&'a String> for SmolStr {
    fn from_iter<I: iter::IntoIterator<Item = &'a String>>(iter: I) -> SmolStr {
        SmolStr::from_iter(iter.into_iter().map(|x| x.as_str()))
    }
}

impl<'a> iter::FromIterator<&'a str> for SmolStr {
    fn from_iter<I: iter::IntoIterator<Item = &'a str>>(iter: I) -> SmolStr {
        build_from_str_iter(iter.into_iter())
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
        text.as_str().into()
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
        T: AsRef<str>,
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

            if len <= N_NEWLINES + N_SPACES {
                let bytes = text.as_bytes();
                let possible_newline_count = cmp::min(len, N_NEWLINES);
                let newlines = bytes[..possible_newline_count]
                    .iter()
                    .take_while(|&&b| b == b'\n')
                    .count();
                let possible_space_count = len - newlines;
                if possible_space_count <= N_SPACES && bytes[newlines..].iter().all(|&b| b == b' ')
                {
                    let spaces = possible_space_count;
                    return Repr::Substring { newlines, spaces };
                }
            }
        }

        Repr::Heap(text.as_ref().into())
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

    #[inline]
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
    use super::SmolStr;
    use ::serde::de::{Deserializer, Error, Unexpected, Visitor};
    use std::fmt;

    // https://github.com/serde-rs/serde/blob/629802f2abfd1a54a6072992888fea7ca5bc209f/serde/src/private/de.rs#L56-L125
    fn smol_str<'de: 'a, 'a, D>(deserializer: D) -> Result<SmolStr, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SmolStrVisitor;

        impl<'a> Visitor<'a> for SmolStrVisitor {
            type Value = SmolStr;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a string")
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(SmolStr::from(v))
            }

            fn visit_borrowed_str<E>(self, v: &'a str) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(SmolStr::from(v))
            }

            fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(SmolStr::from(v))
            }

            fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
            where
                E: Error,
            {
                match std::str::from_utf8(v) {
                    Ok(s) => Ok(SmolStr::from(s)),
                    Err(_) => Err(Error::invalid_value(Unexpected::Bytes(v), &self)),
                }
            }

            fn visit_borrowed_bytes<E>(self, v: &'a [u8]) -> Result<Self::Value, E>
            where
                E: Error,
            {
                match std::str::from_utf8(v) {
                    Ok(s) => Ok(SmolStr::from(s)),
                    Err(_) => Err(Error::invalid_value(Unexpected::Bytes(v), &self)),
                }
            }

            fn visit_byte_buf<E>(self, v: Vec<u8>) -> Result<Self::Value, E>
            where
                E: Error,
            {
                match String::from_utf8(v) {
                    Ok(s) => Ok(SmolStr::from(s)),
                    Err(e) => Err(Error::invalid_value(
                        Unexpected::Bytes(&e.into_bytes()),
                        &self,
                    )),
                }
            }
        }

        deserializer.deserialize_str(SmolStrVisitor)
    }

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
            smol_str(deserializer)
        }
    }
}
