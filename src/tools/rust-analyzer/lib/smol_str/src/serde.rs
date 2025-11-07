use alloc::{string::String, vec::Vec};
use core::fmt;

use serde::de::{Deserializer, Error, Unexpected, Visitor};
use serde_core as serde;

use crate::SmolStr;

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
            match core::str::from_utf8(v) {
                Ok(s) => Ok(SmolStr::from(s)),
                Err(_) => Err(Error::invalid_value(Unexpected::Bytes(v), &self)),
            }
        }

        fn visit_borrowed_bytes<E>(self, v: &'a [u8]) -> Result<Self::Value, E>
        where
            E: Error,
        {
            match core::str::from_utf8(v) {
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
                Err(e) => Err(Error::invalid_value(Unexpected::Bytes(&e.into_bytes()), &self)),
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
