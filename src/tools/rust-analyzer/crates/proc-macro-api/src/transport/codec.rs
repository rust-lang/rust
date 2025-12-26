//! Protocol codec

use std::io;

use serde::de::DeserializeOwned;

use crate::transport::framing::Framing;

pub mod json;
pub mod postcard;

pub trait Codec: Framing {
    fn encode<T: serde::Serialize>(msg: &T) -> io::Result<Self::Buf>;
    fn decode<T: DeserializeOwned>(buf: &mut Self::Buf) -> io::Result<T>;
}
