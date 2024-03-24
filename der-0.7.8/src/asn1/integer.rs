//! ASN.1 `INTEGER` support.

pub(super) mod int;
pub(super) mod uint;

use core::{cmp::Ordering, mem};

use crate::{EncodeValue, Result, SliceWriter};

/// Is the highest bit of the first byte in the slice set to `1`? (if present)
#[inline]
fn is_highest_bit_set(bytes: &[u8]) -> bool {
    bytes
        .first()
        .map(|byte| byte & 0b10000000 != 0)
        .unwrap_or(false)
}

/// Compare two integer values
fn value_cmp<T>(a: T, b: T) -> Result<Ordering>
where
    T: Copy + EncodeValue + Sized,
{
    const MAX_INT_SIZE: usize = 16;
    debug_assert!(mem::size_of::<T>() <= MAX_INT_SIZE);

    let mut buf1 = [0u8; MAX_INT_SIZE];
    let mut encoder1 = SliceWriter::new(&mut buf1);
    a.encode_value(&mut encoder1)?;

    let mut buf2 = [0u8; MAX_INT_SIZE];
    let mut encoder2 = SliceWriter::new(&mut buf2);
    b.encode_value(&mut encoder2)?;

    Ok(encoder1.finish()?.cmp(encoder2.finish()?))
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::{Decode, Encode};

    // Vectors from Section 5.7 of:
    // https://luca.ntop.org/Teaching/Appunti/asn1.html
    pub(crate) const I0_BYTES: &[u8] = &[0x02, 0x01, 0x00];
    pub(crate) const I127_BYTES: &[u8] = &[0x02, 0x01, 0x7F];
    pub(crate) const I128_BYTES: &[u8] = &[0x02, 0x02, 0x00, 0x80];
    pub(crate) const I256_BYTES: &[u8] = &[0x02, 0x02, 0x01, 0x00];
    pub(crate) const INEG128_BYTES: &[u8] = &[0x02, 0x01, 0x80];
    pub(crate) const INEG129_BYTES: &[u8] = &[0x02, 0x02, 0xFF, 0x7F];

    // Additional vectors
    pub(crate) const I255_BYTES: &[u8] = &[0x02, 0x02, 0x00, 0xFF];
    pub(crate) const I32767_BYTES: &[u8] = &[0x02, 0x02, 0x7F, 0xFF];
    pub(crate) const I65535_BYTES: &[u8] = &[0x02, 0x03, 0x00, 0xFF, 0xFF];
    pub(crate) const INEG32768_BYTES: &[u8] = &[0x02, 0x02, 0x80, 0x00];

    #[test]
    fn decode_i8() {
        assert_eq!(0, i8::from_der(I0_BYTES).unwrap());
        assert_eq!(127, i8::from_der(I127_BYTES).unwrap());
        assert_eq!(-128, i8::from_der(INEG128_BYTES).unwrap());
    }

    #[test]
    fn decode_i16() {
        assert_eq!(0, i16::from_der(I0_BYTES).unwrap());
        assert_eq!(127, i16::from_der(I127_BYTES).unwrap());
        assert_eq!(128, i16::from_der(I128_BYTES).unwrap());
        assert_eq!(255, i16::from_der(I255_BYTES).unwrap());
        assert_eq!(256, i16::from_der(I256_BYTES).unwrap());
        assert_eq!(32767, i16::from_der(I32767_BYTES).unwrap());
        assert_eq!(-128, i16::from_der(INEG128_BYTES).unwrap());
        assert_eq!(-129, i16::from_der(INEG129_BYTES).unwrap());
        assert_eq!(-32768, i16::from_der(INEG32768_BYTES).unwrap());
    }

    #[test]
    fn decode_u8() {
        assert_eq!(0, u8::from_der(I0_BYTES).unwrap());
        assert_eq!(127, u8::from_der(I127_BYTES).unwrap());
        assert_eq!(255, u8::from_der(I255_BYTES).unwrap());
    }

    #[test]
    fn decode_u16() {
        assert_eq!(0, u16::from_der(I0_BYTES).unwrap());
        assert_eq!(127, u16::from_der(I127_BYTES).unwrap());
        assert_eq!(255, u16::from_der(I255_BYTES).unwrap());
        assert_eq!(256, u16::from_der(I256_BYTES).unwrap());
        assert_eq!(32767, u16::from_der(I32767_BYTES).unwrap());
        assert_eq!(65535, u16::from_der(I65535_BYTES).unwrap());
    }

    #[test]
    fn encode_i8() {
        let mut buffer = [0u8; 3];

        assert_eq!(I0_BYTES, 0i8.encode_to_slice(&mut buffer).unwrap());
        assert_eq!(I127_BYTES, 127i8.encode_to_slice(&mut buffer).unwrap());

        assert_eq!(
            INEG128_BYTES,
            (-128i8).encode_to_slice(&mut buffer).unwrap()
        );
    }

    #[test]
    fn encode_i16() {
        let mut buffer = [0u8; 4];
        assert_eq!(I0_BYTES, 0i16.encode_to_slice(&mut buffer).unwrap());
        assert_eq!(I127_BYTES, 127i16.encode_to_slice(&mut buffer).unwrap());
        assert_eq!(I128_BYTES, 128i16.encode_to_slice(&mut buffer).unwrap());
        assert_eq!(I255_BYTES, 255i16.encode_to_slice(&mut buffer).unwrap());
        assert_eq!(I256_BYTES, 256i16.encode_to_slice(&mut buffer).unwrap());
        assert_eq!(I32767_BYTES, 32767i16.encode_to_slice(&mut buffer).unwrap());

        assert_eq!(
            INEG128_BYTES,
            (-128i16).encode_to_slice(&mut buffer).unwrap()
        );

        assert_eq!(
            INEG129_BYTES,
            (-129i16).encode_to_slice(&mut buffer).unwrap()
        );

        assert_eq!(
            INEG32768_BYTES,
            (-32768i16).encode_to_slice(&mut buffer).unwrap()
        );
    }

    #[test]
    fn encode_u8() {
        let mut buffer = [0u8; 4];
        assert_eq!(I0_BYTES, 0u8.encode_to_slice(&mut buffer).unwrap());
        assert_eq!(I127_BYTES, 127u8.encode_to_slice(&mut buffer).unwrap());
        assert_eq!(I255_BYTES, 255u8.encode_to_slice(&mut buffer).unwrap());
    }

    #[test]
    fn encode_u16() {
        let mut buffer = [0u8; 5];
        assert_eq!(I0_BYTES, 0u16.encode_to_slice(&mut buffer).unwrap());
        assert_eq!(I127_BYTES, 127u16.encode_to_slice(&mut buffer).unwrap());
        assert_eq!(I128_BYTES, 128u16.encode_to_slice(&mut buffer).unwrap());
        assert_eq!(I255_BYTES, 255u16.encode_to_slice(&mut buffer).unwrap());
        assert_eq!(I256_BYTES, 256u16.encode_to_slice(&mut buffer).unwrap());
        assert_eq!(I32767_BYTES, 32767u16.encode_to_slice(&mut buffer).unwrap());
        assert_eq!(I65535_BYTES, 65535u16.encode_to_slice(&mut buffer).unwrap());
    }

    /// Integers must be encoded with a minimum number of octets
    #[test]
    fn reject_non_canonical() {
        assert!(i8::from_der(&[0x02, 0x02, 0x00, 0x00]).is_err());
        assert!(i16::from_der(&[0x02, 0x02, 0x00, 0x00]).is_err());
        assert!(u8::from_der(&[0x02, 0x02, 0x00, 0x00]).is_err());
        assert!(u16::from_der(&[0x02, 0x02, 0x00, 0x00]).is_err());
    }
}
