use std::{fmt, ops::Deref, sync::Arc};

/// A `SmolStr` is a string type that has the following properties
///
///  * `size_of::<SmolStr>() == size_of::<String>()`
///  * Strings up to 22 bytes long do not use heap allocations
///  * Runs of `\n` and space symbols (typical whitespace pattern of indentation
///    in programming laguages) do not use heap allocations
///  * `Clone` is `O(1)`
///
/// Unlike `String`, however, `SmolStr` is immutable. The primary use-case for
/// `SmolStr` is a good enough default storage for tokens of typical programming
/// languages. A specialized interner might be a better solution for some use-cases.
///
/// Intenrally, `SmolStr` is roughly an `enum { Heap<Arc<str>>, Inline([u8; 22]) }`.
#[derive(Clone)]
pub struct SmolStr(Repr);

impl SmolStr {
    pub fn new(text: &str) -> SmolStr {
        SmolStr(Repr::new(text))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    pub fn to_string(&self) -> String {
        self.as_str().to_string()
    }
}

impl Deref for SmolStr {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}

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

impl<'a> From<&'a str> for SmolStr {
    fn from(text: &'a str) -> Self {
        Self::new(text)
    }
}

const INLINE_CAP: usize = 22;
const WS_TAG: u8 = (INLINE_CAP + 1) as u8;
const N_NEWLINES: usize = 32;
const N_SPACES: usize = 128;
const WS: &str =
    "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n                                                                                                                                ";

#[derive(Clone, Debug)]
enum Repr {
    Heap(Arc<str>),
    Inline { len: u8, buf: [u8; INLINE_CAP] },
}

impl Repr {
    fn new(text: &str) -> Repr {
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
            let mut buf = [0; INLINE_CAP];
            buf[0] = newlines as u8;
            buf[1] = spaces as u8;
            return Repr::Inline { len: WS_TAG, buf };
        }

        Repr::Heap(text.to_string().into_boxed_str().into())
    }

    fn as_str(&self) -> &str {
        match self {
            Repr::Heap(data) => &*data,
            Repr::Inline { len, buf } => {
                if *len == WS_TAG {
                    let newlines = buf[0] as usize;
                    let spaces = buf[1] as usize;
                    assert!(newlines <= N_NEWLINES && spaces <= N_SPACES);
                    return &WS[N_NEWLINES - newlines..N_NEWLINES + spaces];
                }

                let len = *len as usize;
                let buf = &buf[..len];
                unsafe { ::std::str::from_utf8_unchecked(buf) }
            }
        }
    }
}
