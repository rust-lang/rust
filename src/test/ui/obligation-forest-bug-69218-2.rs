// run-pass
#![allow(dead_code)]

use std::borrow::Cow;

pub type Result<T> = std::result::Result<T, &'static str>;

pub struct CompressedData<'data> {
    pub format: CompressionFormat,
    pub data: &'data [u8],
}

pub enum CompressionFormat {
    None,
    Unknown,
}

impl<'data> CompressedData<'data> {
    pub fn decompress(self) -> Result<Cow<'data, [u8]>> {
        match self.format {
            CompressionFormat::None => Ok(Cow::Borrowed(self.data)),
            _ => Err("Unsupported compressed data."),
        }
    }
}

fn main() {}
