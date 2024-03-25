//! ASN.1 `REAL` support.

// TODO(tarcieri): checked arithmetic
#![allow(
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::integer_arithmetic
)]

use crate::{
    BytesRef, DecodeValue, EncodeValue, FixedTag, Header, Length, Reader, Result, StrRef, Tag,
    Writer,
};

use super::integer::uint::strip_leading_zeroes;

impl<'a> DecodeValue<'a> for f64 {
    fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
        let bytes = BytesRef::decode_value(reader, header)?.as_slice();

        if header.length == Length::ZERO {
            Ok(0.0)
        } else if is_nth_bit_one::<7>(bytes) {
            // Binary encoding from section 8.5.7 applies
            let sign: u64 = u64::from(is_nth_bit_one::<6>(bytes));

            // Section 8.5.7.2: Check the base -- the DER specs say that only base 2 should be supported in DER
            let base = mnth_bits_to_u8::<5, 4>(bytes);

            if base != 0 {
                // Real related error: base is not DER compliant (base encoded in enum)
                return Err(Tag::Real.value_error());
            }

            // Section 8.5.7.3
            let scaling_factor = mnth_bits_to_u8::<3, 2>(bytes);

            // Section 8.5.7.4
            let mantissa_start;
            let exponent = match mnth_bits_to_u8::<1, 0>(bytes) {
                0 => {
                    mantissa_start = 2;
                    let ebytes = (i16::from_be_bytes([0x0, bytes[1]])).to_be_bytes();
                    u64::from_be_bytes([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, ebytes[0], ebytes[1]])
                }
                1 => {
                    mantissa_start = 3;
                    let ebytes = (i16::from_be_bytes([bytes[1], bytes[2]])).to_be_bytes();
                    u64::from_be_bytes([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, ebytes[0], ebytes[1]])
                }
                _ => {
                    // Real related error: encoded exponent cannot be represented on an IEEE-754 double
                    return Err(Tag::Real.value_error());
                }
            };
            // Section 8.5.7.5: Read the remaining bytes for the mantissa
            let mut n_bytes = [0x0; 8];
            for (pos, byte) in bytes[mantissa_start..].iter().rev().enumerate() {
                n_bytes[7 - pos] = *byte;
            }
            let n = u64::from_be_bytes(n_bytes);
            // Multiply byt 2^F corresponds to just a left shift
            let mantissa = n << scaling_factor;
            // Create the f64
            Ok(encode_f64(sign, exponent, mantissa))
        } else if is_nth_bit_one::<6>(bytes) {
            // This either a special value, or it's the value minus zero is encoded, section 8.5.9 applies
            match mnth_bits_to_u8::<1, 0>(bytes) {
                0 => Ok(f64::INFINITY),
                1 => Ok(f64::NEG_INFINITY),
                2 => Ok(f64::NAN),
                3 => Ok(-0.0_f64),
                _ => Err(Tag::Real.value_error()),
            }
        } else {
            let astr = StrRef::from_bytes(&bytes[1..])?;
            match astr.inner.parse::<f64>() {
                Ok(val) => Ok(val),
                // Real related error: encoding not supported or malformed
                Err(_) => Err(Tag::Real.value_error()),
            }
        }
    }
}

impl EncodeValue for f64 {
    fn value_len(&self) -> Result<Length> {
        if self.is_sign_positive() && (*self) < f64::MIN_POSITIVE {
            // Zero: positive yet smaller than the minimum positive number
            Ok(Length::ZERO)
        } else if self.is_nan()
            || self.is_infinite()
            || (self.is_sign_negative() && -self < f64::MIN_POSITIVE)
        {
            // NaN, infinite (positive or negative), or negative zero (negative but its negative is less than the min positive number)
            Ok(Length::ONE)
        } else {
            // The length is that of the first octets plus those needed for the exponent plus those needed for the mantissa
            let (_sign, exponent, mantissa) = decode_f64(*self);

            let exponent_len = if exponent == 0 {
                // Section 8.5.7.4: there must be at least one octet for exponent encoding
                // But, if the exponent is zero, it'll be skipped, so we make sure force it to 1
                Length::ONE
            } else {
                let ebytes = exponent.to_be_bytes();
                Length::try_from(strip_leading_zeroes(&ebytes).len())?
            };

            let mantissa_len = if mantissa == 0 {
                Length::ONE
            } else {
                let mbytes = mantissa.to_be_bytes();
                Length::try_from(strip_leading_zeroes(&mbytes).len())?
            };

            exponent_len + mantissa_len + Length::ONE
        }
    }

    fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
        // Check if special value
        // Encode zero first, if it's zero
        // Special value from section 8.5.9 if non zero
        if self.is_nan()
            || self.is_infinite()
            || (self.is_sign_negative() && -self < f64::MIN_POSITIVE)
            || (self.is_sign_positive() && (*self) < f64::MIN_POSITIVE)
        {
            if self.is_sign_positive() && (*self) < f64::MIN_POSITIVE {
                // Zero
                return Ok(());
            } else if self.is_nan() {
                // Not a number
                writer.write_byte(0b0100_0010)?;
            } else if self.is_infinite() {
                if self.is_sign_negative() {
                    // Negative infinity
                    writer.write_byte(0b0100_0001)?;
                } else {
                    // Plus infinity
                    writer.write_byte(0b0100_0000)?;
                }
            } else {
                // Minus zero
                writer.write_byte(0b0100_0011)?;
            }
        } else {
            // Always use binary encoding, set bit 8 to 1
            let mut first_byte = 0b1000_0000;

            if self.is_sign_negative() {
                // Section 8.5.7.1: set bit 7 to 1 if negative
                first_byte |= 0b0100_0000;
            }

            // Bits 6 and 5 are set to 0 to specify that binary encoding is used
            //
            // NOTE: the scaling factor is only used to align the implicit point of the mantissa.
            // This is unnecessary in DER because the base is 2, and therefore necessarily aligned.
            // Therefore, we do not modify the mantissa in anyway after this function call, which
            // already adds the implicit one of the IEEE 754 representation.
            let (_sign, exponent, mantissa) = decode_f64(*self);

            // Encode the exponent as two's complement on 16 bits and remove the bias
            let exponent_bytes = exponent.to_be_bytes();
            let ebytes = strip_leading_zeroes(&exponent_bytes);

            match ebytes.len() {
                0 | 1 => {}
                2 => first_byte |= 0b0000_0001,
                3 => first_byte |= 0b0000_0010,
                _ => {
                    // TODO: support multi octet exponent encoding?
                    return Err(Tag::Real.value_error());
                }
            }

            writer.write_byte(first_byte)?;

            // Encode both bytes or just the last one, handled by encode_bytes directly
            // Rust already encodes the data as two's complement, so no further processing is needed
            writer.write(ebytes)?;

            // Now, encode the mantissa as unsigned binary number
            let mantissa_bytes = mantissa.to_be_bytes();
            let mbytes = strip_leading_zeroes(&mantissa_bytes);
            writer.write(mbytes)?;
        }

        Ok(())
    }
}

impl FixedTag for f64 {
    const TAG: Tag = Tag::Real;
}

/// Is the N-th bit 1 in the first octet?
/// NOTE: this function is zero indexed
pub(crate) fn is_nth_bit_one<const N: usize>(bytes: &[u8]) -> bool {
    if N < 8 {
        bytes
            .first()
            .map(|byte| byte & (1 << N) != 0)
            .unwrap_or(false)
    } else {
        false
    }
}

/// Convert bits M, N into a u8, in the first octet only
pub(crate) fn mnth_bits_to_u8<const M: usize, const N: usize>(bytes: &[u8]) -> u8 {
    let bit_m = is_nth_bit_one::<M>(bytes);
    let bit_n = is_nth_bit_one::<N>(bytes);
    (bit_m as u8) << 1 | bit_n as u8
}

/// Decode an f64 as its sign, exponent, and mantissa in u64 and in that order, using bit shifts and masks.
/// Note: this function **removes** the 1023 bias from the exponent and adds the implicit 1
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn decode_f64(f: f64) -> (u64, u64, u64) {
    let bits = f.to_bits();
    let sign = bits >> 63;
    let exponent = bits >> 52 & 0x7ff;
    let exponent_bytes_no_bias = (exponent as i16 - 1023).to_be_bytes();
    let exponent_no_bias = u64::from_be_bytes([
        0x0,
        0x0,
        0x0,
        0x0,
        0x0,
        0x0,
        exponent_bytes_no_bias[0],
        exponent_bytes_no_bias[1],
    ]);
    let mantissa = bits & 0xfffffffffffff;
    (sign, exponent_no_bias, mantissa + 1)
}

/// Encode an f64 from its sign, exponent (**without** the 1023 bias), and (mantissa - 1) using bit shifts as received by ASN1
pub(crate) fn encode_f64(sign: u64, exponent: u64, mantissa: u64) -> f64 {
    // Add the bias to the exponent
    let exponent_with_bias =
        (i16::from_be_bytes([exponent.to_be_bytes()[6], exponent.to_be_bytes()[7]]) + 1023) as u64;
    let bits = sign << 63 | exponent_with_bias << 52 | (mantissa - 1);
    f64::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use crate::{Decode, Encode};

    #[test]
    fn decode_subnormal() {
        assert!(f64::from_der(&[0x09, 0x01, 0b0100_0010]).unwrap().is_nan());
        let plus_infty = f64::from_der(&[0x09, 0x01, 0b0100_0000]).unwrap();
        assert!(plus_infty.is_infinite() && plus_infty.is_sign_positive());
        let neg_infty = f64::from_der(&[0x09, 0x01, 0b0100_0001]).unwrap();
        assert!(neg_infty.is_infinite() && neg_infty.is_sign_negative());
        let neg_zero = f64::from_der(&[0x09, 0x01, 0b0100_0011]).unwrap();
        assert!(neg_zero.is_sign_negative() && neg_zero.abs() < f64::EPSILON);
    }

    #[test]
    fn encode_subnormal() {
        // All subnormal fit in three bytes
        let mut buffer = [0u8; 3];
        assert_eq!(
            &[0x09, 0x01, 0b0100_0010],
            f64::NAN.encode_to_slice(&mut buffer).unwrap()
        );
        assert_eq!(
            &[0x09, 0x01, 0b0100_0000],
            f64::INFINITY.encode_to_slice(&mut buffer).unwrap()
        );
        assert_eq!(
            &[0x09, 0x01, 0b0100_0001],
            f64::NEG_INFINITY.encode_to_slice(&mut buffer).unwrap()
        );
        assert_eq!(
            &[0x09, 0x01, 0b0100_0011],
            (-0.0_f64).encode_to_slice(&mut buffer).unwrap()
        );
    }

    #[test]
    fn encdec_normal() {
        // The comments correspond to the decoded value from the ASN.1 playground when the bytes are inputed.
        {
            // rec1value R ::= 0
            let val = 0.0;
            let expected = &[0x09, 0x0];
            let mut buffer = [0u8; 2];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }

        {
            // rec1value R ::= { mantissa 1, base 2, exponent 0 }
            let val = 1.0;
            let expected = &[0x09, 0x03, 0x80, 0x00, 0x01];
            let mut buffer = [0u8; 5];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }

        {
            // rec1value R ::= { mantissa -1, base 2, exponent 0 }
            let val = -1.0;
            let expected = &[0x09, 0x03, 0xc0, 0x00, 0x01];
            let mut buffer = [0u8; 5];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }

        {
            // rec1value R ::= { mantissa -1, base 2, exponent 1 }
            let val = -1.0000000000000002;
            let expected = &[0x09, 0x03, 0xc0, 0x00, 0x02];
            let mut buffer = [0u8; 5];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }

        {
            // rec1value R ::= { mantissa 1, base 2, exponent -1022 }
            // NOTE: f64::MIN_EXP == -1021 so the exponent decoded by ASN.1 is what we expect
            let val = f64::MIN_POSITIVE;
            let expected = &[0x09, 0x04, 0x81, 0xfc, 0x02, 0x01];
            let mut buffer = [0u8; 7];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }

        {
            // rec4value R ::= { mantissa 1, base 2, exponent 3 }
            let val = 1.0000000000000016;
            let expected = &[0x09, 0x03, 0x80, 0x00, 0x08];
            let mut buffer = [0u8; 5];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }

        {
            // rec5value R ::= { mantissa 4222124650659841, base 2, exponent 4 }
            let val = 31.0;
            let expected = &[
                0x9, 0x9, 0x80, 0x04, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
            ];
            let mut buffer = [0u8; 11];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }
    }

    #[test]
    fn encdec_irrationals() {
        {
            let val = core::f64::consts::PI;
            let expected = &[
                0x09, 0x09, 0x80, 0x01, 0x09, 0x21, 0xfb, 0x54, 0x44, 0x2d, 0x19,
            ];
            let mut buffer = [0u8; 11];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }

        {
            let val = core::f64::consts::E;
            let expected = &[
                0x09, 0x09, 0x80, 0x01, 0x05, 0xbf, 0x0a, 0x8b, 0x14, 0x57, 0x6a,
            ];
            let mut buffer = [0u8; 12];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }
        {
            let val = core::f64::consts::LN_2;
            let expected = &[
                0x09, 0x0a, 0x81, 0xff, 0xff, 0x6, 0x2e, 0x42, 0xfe, 0xfa, 0x39, 0xf0,
            ];
            let mut buffer = [0u8; 12];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }
    }

    #[test]
    fn encdec_reasonable_f64() {
        // Tests the encoding and decoding of reals with some arbitrary numbers
        {
            // rec1value R ::= { mantissa 2414341043715239, base 2, exponent 21 }
            let val = 3221417.1584163485;
            let expected = &[
                0x9, 0x9, 0x80, 0x15, 0x8, 0x93, 0xd4, 0x94, 0x46, 0xfc, 0xa7,
            ];
            let mut buffer = [0u8; 11];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }

        {
            // rec1value R ::= { mantissa 2671155248072715, base 2, exponent 23 }
            let val = 13364022.365665454;
            let expected = &[
                0x09, 0x09, 0x80, 0x17, 0x09, 0x7d, 0x66, 0xcb, 0xb3, 0x88, 0x0b,
            ];
            let mut buffer = [0u8; 12];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }

        {
            // rec1value R ::= { mantissa -4386812962460287, base 2, exponent 14 }
            let val = -32343.132588105735;
            let expected = &[
                0x09, 0x09, 0xc0, 0x0e, 0x0f, 0x95, 0xc8, 0x7c, 0x52, 0xd2, 0x7f,
            ];
            let mut buffer = [0u8; 12];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }

        {
            let val = -27084.866751869475;
            let expected = &[
                0x09, 0x09, 0xc0, 0x0e, 0x0a, 0x73, 0x37, 0x78, 0xdc, 0xd5, 0x4a,
            ];
            let mut buffer = [0u8; 12];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }

        {
            // rec1value R ::= { mantissa -4372913134428149, base 2, exponent 7 }
            let val = -252.28566647111404;
            let expected = &[
                0x09, 0x09, 0xc0, 0x07, 0x0f, 0x89, 0x24, 0x2e, 0x02, 0xdf, 0xf5,
            ];
            let mut buffer = [0u8; 12];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }

        {
            let val = -14.399709612928548;
            let expected = &[
                0x09, 0x09, 0xc0, 0x03, 0x0c, 0xcc, 0xa6, 0xbd, 0x06, 0xd9, 0x92,
            ];
            let mut buffer = [0u8; 12];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }

        {
            let val = -0.08340570261832964;
            let expected = &[
                0x09, 0x0a, 0xc1, 0xff, 0xfc, 0x05, 0x5a, 0x13, 0x7d, 0x0b, 0xae, 0x3d,
            ];
            let mut buffer = [0u8; 12];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }

        {
            let val = 0.00536851453803701;
            let expected = &[
                0x09, 0x0a, 0x81, 0xff, 0xf8, 0x05, 0xfd, 0x4b, 0xa5, 0xe7, 0x4c, 0x93,
            ];
            let mut buffer = [0u8; 12];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }

        {
            let val = 0.00045183525648866433;
            let expected = &[
                0x09, 0x0a, 0x81, 0xff, 0xf4, 0x0d, 0x9c, 0x89, 0xa6, 0x59, 0x33, 0x39,
            ];
            let mut buffer = [0u8; 12];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }

        {
            let val = 0.000033869092002682955;
            let expected = &[
                0x09, 0x0a, 0x81, 0xff, 0xf1, 0x01, 0xc1, 0xd5, 0x23, 0xd5, 0x54, 0x7c,
            ];
            let mut buffer = [0u8; 12];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }

        {
            let val = 0.0000011770891033600088;
            let expected = &[
                0x09, 0x0a, 0x81, 0xff, 0xec, 0x03, 0xbf, 0x8f, 0x27, 0xf4, 0x62, 0x56,
            ];
            let mut buffer = [0u8; 12];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }

        {
            let val = 0.00000005549514041997082;
            let expected = &[
                0x09, 0x0a, 0x81, 0xff, 0xe7, 0x0d, 0xcb, 0x31, 0xab, 0x6e, 0xb8, 0xd7,
            ];
            let mut buffer = [0u8; 12];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }

        {
            let val = 0.0000000012707044685547803;
            let expected = &[
                0x09, 0x0a, 0x81, 0xff, 0xe2, 0x05, 0xd4, 0x9e, 0x0a, 0xf2, 0xff, 0x1f,
            ];
            let mut buffer = [0u8; 12];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }

        {
            let val = 0.00000000002969611878378562;
            let expected = &[
                0x09, 0x09, 0x81, 0xff, 0xdd, 0x53, 0x5b, 0x6f, 0x97, 0xee, 0xb6,
            ];
            let mut buffer = [0u8; 11];
            let encoded = val.encode_to_slice(&mut buffer).unwrap();
            assert_eq!(
                expected, encoded,
                "invalid encoding of {}:\ngot  {:x?}\nwant: {:x?}",
                val, encoded, expected
            );
            let decoded = f64::from_der(encoded).unwrap();
            assert!(
                (decoded - val).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                val,
                decoded
            );
        }
    }

    #[test]
    fn reject_non_canonical() {
        assert!(f64::from_der(&[0x09, 0x81, 0x00]).is_err());
    }

    #[test]
    fn encdec_f64() {
        use super::{decode_f64, encode_f64};
        // Test that the extraction and recreation works
        for val in [
            1.0,
            0.1,
            -0.1,
            -1.0,
            0.0,
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::MIN,
            3.1415,
            951.2357864,
            -3.1415,
            -951.2357864,
        ] {
            let (s, e, m) = decode_f64(val);
            let val2 = encode_f64(s, e, m);
            assert!(
                (val - val2).abs() < f64::EPSILON,
                "fail - want {}\tgot {}",
                val,
                val2
            );
        }
    }

    #[test]
    fn validation_cases() {
        // Caveat: these test cases are validated on the ASN.1 playground: https://asn1.io/asn1playground/ .
        // The test case consists in inputing the bytes in the "decode" field and checking that the decoded
        // value corresponds to the one encoded here.
        // This tool encodes _all_ values that are non-zero in the ISO 6093 NR3 representation.
        // This does not seem to perfectly adhere to the ITU specifications, Special Cases section.
        // The implementation of this crate correctly supports decoding such values. It will, however,
        // systematically encode REALs in their base 2 form, with a scaling factor where needed to
        // ensure that the mantissa is either odd or zero (as per section 11.3.1).

        // Positive trivial numbers
        {
            let expect = 10.0;
            let testcase = &[0x09, 0x05, 0x03, 0x31, 0x2E, 0x45, 0x31];
            let decoded = f64::from_der(testcase).unwrap();
            assert!(
                (decoded - expect).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                expect,
                decoded
            );
        }
        {
            let expect = 100.0;
            let testcase = &[0x09, 0x05, 0x03, 0x31, 0x2E, 0x45, 0x32];
            let decoded = f64::from_der(testcase).unwrap();
            assert!(
                (decoded - expect).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                expect,
                decoded
            );
        }
        {
            let expect = 101.0;
            let testcase = &[0x09, 0x08, 0x03, 0x31, 0x30, 0x31, 0x2E, 0x45, 0x2B, 0x30];
            let decoded = f64::from_der(testcase).unwrap();
            assert!(
                (decoded - expect).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                expect,
                decoded
            );
        }
        {
            let expect = 101.0;
            let testcase = &[0x09, 0x08, 0x03, 0x31, 0x30, 0x31, 0x2E, 0x45, 0x2B, 0x30];
            let decoded = f64::from_der(testcase).unwrap();
            assert!(
                (decoded - expect).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                expect,
                decoded
            );
        }
        {
            let expect = 0.0;
            let testcase = &[0x09, 0x00];
            let decoded = f64::from_der(testcase).unwrap();
            assert!(
                (decoded - expect).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                expect,
                decoded
            );
        }
        {
            let expect = 951.2357864;
            let testcase = &[
                0x09, 0x0F, 0x03, 0x39, 0x35, 0x31, 0x32, 0x33, 0x35, 0x37, 0x38, 0x36, 0x34, 0x2E,
                0x45, 0x2D, 0x37,
            ];
            let decoded = f64::from_der(testcase).unwrap();
            assert!(
                (decoded - expect).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                expect,
                decoded
            );
        }
        // Negative trivial numbers
        {
            let expect = -10.0;
            let testcase = &[0x09, 0x06, 0x03, 0x2D, 0x31, 0x2E, 0x45, 0x31];
            let decoded = f64::from_der(testcase).unwrap();
            assert!(
                (decoded - expect).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                expect,
                decoded
            );
        }
        {
            let expect = -100.0;
            let testcase = &[0x09, 0x06, 0x03, 0x2D, 0x31, 0x2E, 0x45, 0x32];
            let decoded = f64::from_der(testcase).unwrap();
            assert!(
                (decoded - expect).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                expect,
                decoded
            );
        }
        {
            let expect = -101.0;
            let testcase = &[
                0x09, 0x09, 0x03, 0x2D, 0x31, 0x30, 0x31, 0x2E, 0x45, 0x2B, 0x30,
            ];
            let decoded = f64::from_der(testcase).unwrap();
            assert!(
                (decoded - expect).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                expect,
                decoded
            );
        }
        {
            let expect = -0.5;
            let testcase = &[0x09, 0x07, 0x03, 0x2D, 0x35, 0x2E, 0x45, 0x2D, 0x31];
            let decoded = f64::from_der(testcase).unwrap();
            assert!(
                (decoded - expect).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                expect,
                decoded
            );
        }
        {
            let expect = -0.0;
            let testcase = &[0x09, 0x03, 0x01, 0x2D, 0x30];
            let decoded = f64::from_der(testcase).unwrap();
            assert!(
                (decoded - expect).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                expect,
                decoded
            );
        }
        {
            // Test NR3 decoding
            let expect = -951.2357864;
            let testcase = &[
                0x09, 0x10, 0x03, 0x2D, 0x39, 0x35, 0x31, 0x32, 0x33, 0x35, 0x37, 0x38, 0x36, 0x34,
                0x2E, 0x45, 0x2D, 0x37,
            ];
            let decoded = f64::from_der(testcase).unwrap();
            assert!(
                (decoded - expect).abs() < f64::EPSILON,
                "wanted: {}\tgot: {}",
                expect,
                decoded
            );
        }
    }
}
