use std::{sync::Arc};

const INLINE_CAP: usize = 22;
const WS_TAG: u8 = (INLINE_CAP + 1) as u8;

#[derive(Clone, Debug)]
pub(crate) enum SmolStr {
    Heap(Arc<str>),
    Inline {
        len: u8,
        buf: [u8; INLINE_CAP],
    },
}

impl SmolStr {
    pub fn new(text: &str) -> SmolStr {
        let len = text.len();
        if len <= INLINE_CAP {
            let mut buf = [0; INLINE_CAP];
            buf[..len].copy_from_slice(text.as_bytes());
            return SmolStr::Inline { len: len as u8, buf };
        }

        let newlines = text.bytes().take_while(|&b| b == b'\n').count();
        let spaces = text[newlines..].bytes().take_while(|&b| b == b' ').count();
        if newlines + spaces == len && newlines <= N_NEWLINES && spaces <= N_SPACES {
            let mut buf = [0; INLINE_CAP];
            buf[0] = newlines as u8;
            buf[1] = spaces as u8;
            return SmolStr::Inline { len: WS_TAG, buf };
        }

        SmolStr::Heap(
            text.to_string().into_boxed_str().into()
        )
    }

    pub fn as_str(&self) -> &str {
        match self {
            SmolStr::Heap(data) => &*data,
            SmolStr::Inline { len, buf } => {
                if *len == WS_TAG {
                    let newlines = buf[0] as usize;
                    let spaces = buf[1] as usize;
                    assert!(newlines <= N_NEWLINES && spaces <= N_SPACES);
                    return &WS[N_NEWLINES - newlines..N_NEWLINES + spaces]
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

