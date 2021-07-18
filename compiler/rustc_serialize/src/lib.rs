//! Support code for encoding and decoding types.

/*
Core encoding and decoding interfaces.
*/

#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    html_playground_url = "https://play.rust-lang.org/",
    test(attr(allow(unused_variables), deny(warnings)))
)]
#![feature(box_syntax)]
#![feature(never_type)]
#![feature(nll)]
#![feature(associated_type_bounds)]
#![feature(min_specialization)]
#![feature(core_intrinsics)]
#![feature(maybe_uninit_slice)]
#![feature(maybe_uninit_write_slice)]
#![feature(new_uninit)]
#![cfg_attr(test, feature(test))]
#![allow(rustc::internal)]

use std::convert::TryInto;

pub use self::serialize::{Decodable, Decoder, Encodable, Encoder};

mod collection_impls;
mod serialize;

pub mod json;

pub mod leb128;
pub mod opaque;

pub mod raw;

// An integer that will always encode to 8 bytes.
pub struct IntEncodedWithFixedSize(pub u64);

impl IntEncodedWithFixedSize {
    pub const ENCODED_SIZE: usize = 8;
}

impl Encodable<opaque::Encoder> for IntEncodedWithFixedSize {
    #[inline]
    fn encode(&self, e: &mut opaque::Encoder) -> opaque::EncodeResult {
        let _start_pos = e.position();
        e.emit_raw_bytes(&self.0.to_le_bytes())?;
        let _end_pos = e.position();
        debug_assert_eq!((_end_pos - _start_pos), IntEncodedWithFixedSize::ENCODED_SIZE);
        Ok(())
    }
}

impl Encodable<opaque::FileEncoder> for IntEncodedWithFixedSize {
    #[inline]
    fn encode(&self, e: &mut opaque::FileEncoder) -> opaque::FileEncodeResult {
        let _start_pos = e.position();
        e.emit_raw_bytes(&self.0.to_le_bytes())?;
        let _end_pos = e.position();
        debug_assert_eq!((_end_pos - _start_pos), IntEncodedWithFixedSize::ENCODED_SIZE);
        Ok(())
    }
}

impl<'a> Decodable<opaque::Decoder<'a>> for IntEncodedWithFixedSize {
    #[inline]
    fn decode(decoder: &mut opaque::Decoder<'a>) -> Result<IntEncodedWithFixedSize, String> {
        let _start_pos = decoder.position();
        let bytes = decoder.read_raw_bytes(IntEncodedWithFixedSize::ENCODED_SIZE);
        let _end_pos = decoder.position();
        debug_assert_eq!((_end_pos - _start_pos), IntEncodedWithFixedSize::ENCODED_SIZE);

        let value = u64::from_le_bytes(bytes.try_into().unwrap());
        Ok(IntEncodedWithFixedSize(value))
    }
}

impl Encodable<raw::Encoder> for IntEncodedWithFixedSize {
    #[inline]
    fn encode(&self, e: &mut raw::Encoder) -> raw::EncodeResult {
        let _start_pos = e.position();
        e.emit_raw_bytes(&self.0.to_le_bytes())?;
        let _end_pos = e.position();
        debug_assert_eq!((_end_pos - _start_pos), IntEncodedWithFixedSize::ENCODED_SIZE);
        Ok(())
    }
}

impl Encodable<raw::FileEncoder> for IntEncodedWithFixedSize {
    #[inline]
    fn encode(&self, e: &mut raw::FileEncoder) -> raw::FileEncodeResult {
        let _start_pos = e.position();
        e.emit_raw_bytes(&self.0.to_le_bytes())?;
        let _end_pos = e.position();
        debug_assert_eq!((_end_pos - _start_pos), IntEncodedWithFixedSize::ENCODED_SIZE);
        Ok(())
    }
}

impl<'a> Decodable<raw::Decoder<'a>> for IntEncodedWithFixedSize {
    #[inline]
    fn decode(decoder: &mut raw::Decoder<'a>) -> Result<IntEncodedWithFixedSize, String> {
        let _start_pos = decoder.position();
        let bytes = decoder.read_raw_bytes(IntEncodedWithFixedSize::ENCODED_SIZE);
        let _end_pos = decoder.position();
        debug_assert_eq!((_end_pos - _start_pos), IntEncodedWithFixedSize::ENCODED_SIZE);

        let value = u64::from_le_bytes(bytes.try_into().unwrap());
        Ok(IntEncodedWithFixedSize(value))
    }
}
