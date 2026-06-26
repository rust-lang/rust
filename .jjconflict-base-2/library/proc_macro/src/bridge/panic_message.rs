use std::any::Any;

use crate::bridge::{Buffer, Decode, Encode};

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

impl From<PanicMessage> for Box<dyn Any + Send> {
    fn from(val: PanicMessage) -> Self {
        match val {
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

    pub fn into_string(self) -> Option<String> {
        match self {
            PanicMessage::StaticStr(s) => Some(s.into()),
            PanicMessage::String(s) => Some(s),
            PanicMessage::Unknown => None,
        }
    }
}

impl<S> Encode<S> for PanicMessage {
    #[inline]
    fn encode(self, w: &mut Buffer, s: &mut S) {
        self.as_str().encode(w, s);
    }
}

impl<S> Decode<'_, '_, S> for PanicMessage {
    #[inline]
    fn decode(r: &mut &[u8], s: &mut S) -> Self {
        match Option::<String>::decode(r, s) {
            Some(s) => PanicMessage::String(s),
            None => PanicMessage::Unknown,
        }
    }
}
