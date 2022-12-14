//! Serialization for client-server communication.

use std::any::Any;
use std::char;
use std::io::Write;
use std::num::NonZeroU32;
use std::ops::Bound;
use std::str;

pub(super) type Writer = super::buffer::Buffer<u8>;

pub(super) trait Encode<S>: Sized {
    fn encode(self, w: &mut Writer, s: &mut S);
}

pub(super) type Reader<'a> = &'a [u8];

pub(super) trait Decode<'a, 's, S>: Sized {
    fn decode(r: &mut Reader<'a>, s: &'s S) -> Self;
}

pub(super) trait DecodeMut<'a, 's, S>: Sized {
    fn decode(r: &mut Reader<'a>, s: &'s mut S) -> Self;
}

macro_rules! rpc_encode_decode {
    (le $ty:ty) => {
        impl<S> Encode<S> for $ty {
            fn encode(self, w: &mut Writer, _: &mut S) {
                w.extend_from_array(&self.to_le_bytes());
            }
        }

        impl<S> DecodeMut<'_, '_, S> for $ty {
            fn decode(r: &mut Reader<'_>, _: &mut S) -> Self {
                const N: usize = ::std::mem::size_of::<$ty>();

                let mut bytes = [0; N];
                bytes.copy_from_slice(&r[..N]);
                *r = &r[N..];

                Self::from_le_bytes(bytes)
            }
        }
    };
    (struct $name:ident { $($field:ident),* $(,)? }) => {
        impl<S> Encode<S> for $name {
            fn encode(self, w: &mut Writer, s: &mut S) {
                $(self.$field.encode(w, s);)*
            }
        }

        impl<S> DecodeMut<'_, '_, S> for $name {
            fn decode(r: &mut Reader<'_>, s: &mut S) -> Self {
                $name {
                    $($field: DecodeMut::decode(r, s)),*
                }
            }
        }
    };
    (enum $name:ident $(<$($T:ident),+>)? { $($variant:ident $(($field:ident))*),* $(,)? }) => {
        impl<S, $($($T: Encode<S>),+)?> Encode<S> for $name $(<$($T),+>)? {
            fn encode(self, w: &mut Writer, s: &mut S) {
                // HACK(eddyb): `Tag` enum duplicated between the
                // two impls as there's no other place to stash it.
                #[allow(non_upper_case_globals)]
                mod tag {
                    #[repr(u8)] enum Tag { $($variant),* }

                    $(pub const $variant: u8 = Tag::$variant as u8;)*
                }

                match self {
                    $($name::$variant $(($field))* => {
                        tag::$variant.encode(w, s);
                        $($field.encode(w, s);)*
                    })*
                }
            }
        }

        impl<'a, S, $($($T: for<'s> DecodeMut<'a, 's, S>),+)?> DecodeMut<'a, '_, S>
            for $name $(<$($T),+>)?
        {
            fn decode(r: &mut Reader<'a>, s: &mut S) -> Self {
                // HACK(eddyb): `Tag` enum duplicated between the
                // two impls as there's no other place to stash it.
                #[allow(non_upper_case_globals)]
                mod tag {
                    #[repr(u8)] enum Tag { $($variant),* }

                    $(pub const $variant: u8 = Tag::$variant as u8;)*
                }

                match u8::decode(r, s) {
                    $(tag::$variant => {
                        $(let $field = DecodeMut::decode(r, s);)*
                        $name::$variant $(($field))*
                    })*
                    _ => unreachable!(),
                }
            }
        }
    }
}

impl<S> Encode<S> for () {
    fn encode(self, _: &mut Writer, _: &mut S) {}
}

impl<S> DecodeMut<'_, '_, S> for () {
    fn decode(_: &mut Reader<'_>, _: &mut S) -> Self {}
}

impl<S> Encode<S> for u8 {
    fn encode(self, w: &mut Writer, _: &mut S) {
        w.push(self);
    }
}

impl<S> DecodeMut<'_, '_, S> for u8 {
    fn decode(r: &mut Reader<'_>, _: &mut S) -> Self {
        let x = r[0];
        *r = &r[1..];
        x
    }
}

rpc_encode_decode!(le u32);
rpc_encode_decode!(le usize);

impl<S> Encode<S> for bool {
    fn encode(self, w: &mut Writer, s: &mut S) {
        (self as u8).encode(w, s);
    }
}

impl<S> DecodeMut<'_, '_, S> for bool {
    fn decode(r: &mut Reader<'_>, s: &mut S) -> Self {
        match u8::decode(r, s) {
            0 => false,
            1 => true,
            _ => unreachable!(),
        }
    }
}

impl<S> Encode<S> for char {
    fn encode(self, w: &mut Writer, s: &mut S) {
        (self as u32).encode(w, s);
    }
}

impl<S> DecodeMut<'_, '_, S> for char {
    fn decode(r: &mut Reader<'_>, s: &mut S) -> Self {
        char::from_u32(u32::decode(r, s)).unwrap()
    }
}

impl<S> Encode<S> for NonZeroU32 {
    fn encode(self, w: &mut Writer, s: &mut S) {
        self.get().encode(w, s);
    }
}

impl<S> DecodeMut<'_, '_, S> for NonZeroU32 {
    fn decode(r: &mut Reader<'_>, s: &mut S) -> Self {
        Self::new(u32::decode(r, s)).unwrap()
    }
}

impl<S, A: Encode<S>, B: Encode<S>> Encode<S> for (A, B) {
    fn encode(self, w: &mut Writer, s: &mut S) {
        self.0.encode(w, s);
        self.1.encode(w, s);
    }
}

impl<'a, S, A: for<'s> DecodeMut<'a, 's, S>, B: for<'s> DecodeMut<'a, 's, S>> DecodeMut<'a, '_, S>
    for (A, B)
{
    fn decode(r: &mut Reader<'a>, s: &mut S) -> Self {
        (DecodeMut::decode(r, s), DecodeMut::decode(r, s))
    }
}

rpc_encode_decode!(
    enum Bound<T> {
        Included(x),
        Excluded(x),
        Unbounded,
    }
);

rpc_encode_decode!(
    enum Option<T> {
        None,
        Some(x),
    }
);

rpc_encode_decode!(
    enum Result<T, E> {
        Ok(x),
        Err(e),
    }
);

impl<S> Encode<S> for &[u8] {
    fn encode(self, w: &mut Writer, s: &mut S) {
        self.len().encode(w, s);
        w.write_all(self).unwrap();
    }
}

impl<'a, S> DecodeMut<'a, '_, S> for &'a [u8] {
    fn decode(r: &mut Reader<'a>, s: &mut S) -> Self {
        let len = usize::decode(r, s);
        let xs = &r[..len];
        *r = &r[len..];
        xs
    }
}

impl<S> Encode<S> for &str {
    fn encode(self, w: &mut Writer, s: &mut S) {
        self.as_bytes().encode(w, s);
    }
}

impl<'a, S> DecodeMut<'a, '_, S> for &'a str {
    fn decode(r: &mut Reader<'a>, s: &mut S) -> Self {
        str::from_utf8(<&[u8]>::decode(r, s)).unwrap()
    }
}

impl<S> Encode<S> for String {
    fn encode(self, w: &mut Writer, s: &mut S) {
        self[..].encode(w, s);
    }
}

impl<S> DecodeMut<'_, '_, S> for String {
    fn decode(r: &mut Reader<'_>, s: &mut S) -> Self {
        <&str>::decode(r, s).to_string()
    }
}

/// Simplified version of panic payloads, ignoring
/// types other than `&'static str` and `String`.
pub enum PanicMessage {
    StaticStr(&'static str),
    String(String),
    Unknown,
}

impl From<Box<dyn Any + Send>> for PanicMessage {
    fn from(payload: Box<dyn Any + Send + 'static>) -> Self {
        if let Some(s) = payload.downcast_ref::<&'static str>() {
            return PanicMessage::StaticStr(s);
        }
        if let Ok(s) = payload.downcast::<String>() {
            return PanicMessage::String(*s);
        }
        PanicMessage::Unknown
    }
}

impl Into<Box<dyn Any + Send>> for PanicMessage {
    fn into(self) -> Box<dyn Any + Send> {
        match self {
            PanicMessage::StaticStr(s) => Box::new(s),
            PanicMessage::String(s) => Box::new(s),
            PanicMessage::Unknown => {
                struct UnknownPanicMessage;
                Box::new(UnknownPanicMessage)
            }
        }
    }
}

impl PanicMessage {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            PanicMessage::StaticStr(s) => Some(s),
            PanicMessage::String(s) => Some(s),
            PanicMessage::Unknown => None,
        }
    }
}

impl<S> Encode<S> for PanicMessage {
    fn encode(self, w: &mut Writer, s: &mut S) {
        self.as_str().encode(w, s);
    }
}

impl<S> DecodeMut<'_, '_, S> for PanicMessage {
    fn decode(r: &mut Reader<'_>, s: &mut S) -> Self {
        match Option::<String>::decode(r, s) {
            Some(s) => PanicMessage::String(s),
            None => PanicMessage::Unknown,
        }
    }
}
