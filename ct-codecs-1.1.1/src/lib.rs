//! Constant-time codecs.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

mod base64;
mod error;
mod hex;

pub use base64::*;
pub use error::*;
pub use hex::*;

pub trait Encoder {
    /// Length of `bin_len` bytes after encoding.
    fn encoded_len(bin_len: usize) -> Result<usize, Error>;

    /// Encode `bin` into `encoded`.
    /// The output buffer can be larger than required; the returned slice is
    /// a view of the buffer with the correct length.
    fn encode<IN: AsRef<[u8]>>(encoded: &mut [u8], bin: IN) -> Result<&[u8], Error>;

    /// Encode `bin` into `encoded`, return the result as a `str`.
    /// The output buffer can be larger than required; the returned slice is
    /// a view of the buffer with the correct length.
    fn encode_to_str<IN: AsRef<[u8]>>(encoded: &mut [u8], bin: IN) -> Result<&str, Error> {
        Ok(core::str::from_utf8(Self::encode(encoded, bin)?).unwrap())
    }

    /// Encode `bin` as a `String`.
    #[cfg(feature = "std")]
    fn encode_to_string<IN: AsRef<[u8]>>(bin: IN) -> Result<String, Error> {
        let mut encoded = vec![0u8; Self::encoded_len(bin.as_ref().len())?];
        let encoded_len = Self::encode(&mut encoded, bin)?.len();
        encoded.truncate(encoded_len);
        Ok(String::from_utf8(encoded).unwrap())
    }
}

pub trait Decoder {
    /// Decode `encoded` into `bin`.
    /// The output buffer can be larger than required; the returned slice is
    /// a view of the buffer with the correct length.
    /// `ignore` is an optional set of characters to ignore.
    fn decode<'t, IN: AsRef<[u8]>>(
        bin: &'t mut [u8],
        encoded: IN,
        ignore: Option<&[u8]>,
    ) -> Result<&'t [u8], Error>;

    /// Decode `encoded` into a `Vec<u8>`.
    /// `ignore` is an optional set of characters to ignore.
    #[cfg(feature = "std")]
    fn decode_to_vec<IN: AsRef<[u8]>>(
        encoded: IN,
        ignore: Option<&[u8]>,
    ) -> Result<Vec<u8>, Error> {
        let mut bin = vec![0u8; encoded.as_ref().len()];
        let bin_len = Self::decode(&mut bin, encoded, ignore)?.len();
        bin.truncate(bin_len);
        Ok(bin)
    }
}
