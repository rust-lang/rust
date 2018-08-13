use std::{sync::Arc, ops::Deref, fmt};

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

const INLINE_CAP: usize = 22;
const WS_TAG: u8 = (INLINE_CAP + 1) as u8;

#[derive(Clone, Debug)]
enum Repr {
    Heap(Arc<str>),
    Inline {
        len: u8,
        buf: [u8; INLINE_CAP],
    },
}

impl Repr {
    fn new(text: &str) -> Repr {
        let len = text.len();
        if len <= INLINE_CAP {
            let mut buf = [0; INLINE_CAP];
            buf[..len].copy_from_slice(text.as_bytes());
            return Repr::Inline { len: len as u8, buf };
        }

        let newlines = text.bytes().take_while(|&b| b == b'\n').count();
        let spaces = text[newlines..].bytes().take_while(|&b| b == b' ').count();
        if newlines + spaces == len && newlines <= N_NEWLINES && spaces <= N_SPACES {
            let mut buf = [0; INLINE_CAP];
            buf[0] = newlines as u8;
            buf[1] = spaces as u8;
            return Repr::Inline { len: WS_TAG, buf };
        }

        Repr::Heap(
            text.to_string().into_boxed_str().into()
        )
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

const N_NEWLINES: usize = 32;
const N_SPACES: usize = 128;
const WS: &str =
    "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n                                                                                                                                ";


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn smol_str_is_smol() {
        assert_eq!(::std::mem::size_of::<SmolStr>(), 8 + 8 + 8)
    }

    #[test]
    fn test_round_trip() {
        let mut text = String::new();
        for n in 0..256 {
            let smol = SmolStr::new(&text);
            assert_eq!(smol.as_str(), text.as_str());
            text.push_str(&n.to_string());
        }
    }
}

