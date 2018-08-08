use std::{sync::Arc};

const INLINE_CAP: usize = 22;

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
            SmolStr::Inline { len: len as u8, buf }
        } else {
            SmolStr::Heap(
                text.to_string().into_boxed_str().into()
            )
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            SmolStr::Heap(data) => &*data,
            SmolStr::Inline { len, buf } => {
                let len = *len as usize;
                let buf = &buf[..len];
                unsafe { ::std::str::from_utf8_unchecked(buf) }
            }
        }
    }
}

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

