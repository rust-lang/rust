// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
Generate and parse UUIDs

Provides support for Universally Unique Identifiers (UUIDs). A UUID is a
unique 128-bit number, stored as 16 octets.  UUIDs are used to  assign unique
identifiers to entities without requiring a central allocating authority.

They are particularly useful in distributed systems, though can be used in
disparate areas, such as databases and network protocols.  Typically a UUID is
displayed in a readable string form as a sequence of hexadecimals digits,
separated into groups by hyphens.

The uniqueness property is not strictly guaranteed, however for all practical
purposes, it can be assumed that an unintentional collision would be extremely
unlikely.

# Examples

To create a new random (V4) UUID and print it out in hexadecimal form:

```rust
extern mod extra;
use extra::uuid::Uuid;

fn main() {
    let uuid1 = Uuid::new_v4();
    println(uuid1.to_str());
}
 ```

# Strings

Examples of string representations:

* simple: `936DA01F9ABD4d9d80C702AF85C822A8`
* hyphenated: `550e8400-e29b-41d4-a716-446655440000`
* urn: `urn:uuid:F9168C5E-CEB2-4faa-B6BF-329BF39FA1E4`

# References

* [Wikipedia: Universally Unique Identifier](
    http://en.wikipedia.org/wiki/Universally_unique_identifier)
* [RFC4122: A Universally Unique IDentifier (UUID) URN Namespace](
    http://tools.ietf.org/html/rfc4122)

*/

use std::str;
use std::vec;
use std::num::{FromStrRadix, Zero};
use std::char::Char;
use std::container::Container;
use std::to_str::ToStr;
use std::rand;
use std::rand::Rng;
use std::cmp::Eq;
use std::cast::{transmute,transmute_copy};

/// A 128-bit (16 byte) buffer containing the ID
pub type UuidBytes = [u8, ..16];

/// The version of the UUID, denoting the generating algorithm
#[deriving(Eq)]
pub enum UuidVersion {
    /// Version 1: MAC address
    Version1Mac    = 1,
    /// Version 2: DCE Security
    Version2Dce    = 2,
    /// Version 3: MD5 hash
    Version3Md5    = 3,
    /// Version 4: Random
    Version4Random = 4,
    /// Version 5: SHA-1 hash
    Version5Sha1   = 5,
}

/// The reserved variants of UUIDs
#[deriving(Eq)]
pub enum UuidVariant {
    /// Reserved by the NCS for backward compatability
    VariantNCS,
    /// As described in the RFC4122 Specification (default)
    VariantRFC4122,
    /// Resreved by Microsoft for backward compatability
    VariantMicrosoft,
    /// Reserved for future expansion
    VariantFuture,
}

/// A Universally Unique Identifier (UUID)
pub struct Uuid {
    /// The 128-bit number stored in 16 bytes
    bytes: UuidBytes
}

/// A UUID stored as fields (identical to UUID, used only for conversions)
struct UuidFields {
    /// First field, 32-bit word
    data1: u32,
    /// Second field, 16-bit short
    data2: u16,
    /// Third field, 16-bit short
    data3: u16,
    /// Fourth field, 8 bytes
    data4: [u8, ..8]
}

/// Error details for string parsing failures
pub enum ParseError {
    ErrorInvalidLength(uint),
    ErrorInvalidCharacter(char, uint),
    ErrorInvalidGroups(uint),
    ErrorInvalidGroupLength(uint, uint, uint),
}

/// Converts a ParseError to a string
impl ToStr for ParseError {
    #[inline]
    fn to_str(&self) -> ~str {
        match *self {
            ErrorInvalidLength(found) =>
                format!("Invalid length; expecting 32, 36 or 45 chars, found {}",
                        found),
            ErrorInvalidCharacter(found, pos) =>
                format!("Invalid character; found `{}` (0x{:02x}) at offset {}",
                        found, found as uint, pos),
            ErrorInvalidGroups(found) =>
                format!("Malformed; wrong number of groups: expected 1 or 5, found {}",
                        found),
            ErrorInvalidGroupLength(group, found, expecting) =>
                format!("Malformed; length of group {} was {}, expecting {}",
                        group, found, expecting),
        }
    }
}

// Length of each hyphenated group in hex digits
static UuidGroupLens: [uint, ..5] = [8u, 4u, 4u, 4u, 12u];

/// UUID support
impl Uuid {

    /// Returns a nil or empty UUID (containing all zeroes)
    pub fn new_nil() -> Uuid {
        let uuid = Uuid{ bytes: [0, .. 16] };
        uuid
    }

    /// Create a new UUID of the specified version
    pub fn new(v: UuidVersion) -> Option<Uuid> {
        match v {
            Version4Random => Some(Uuid::new_v4()),
            _ => None
        }
    }

    /// Creates a new random UUID
    ///
    /// Uses the `rand` module's default RNG task as the source
    /// of random numbers. Use the rand::Rand trait to supply
    /// a custom generator if required.
    pub fn new_v4() -> Uuid {
        let ub = rand::task_rng().gen_vec(16);
        let mut uuid = Uuid{ bytes: [0, .. 16] };
        vec::bytes::copy_memory(uuid.bytes, ub, 16);
        uuid.set_variant(VariantRFC4122);
        uuid.set_version(Version4Random);
        uuid
    }

    /// Creates a UUID using the supplied field values
    ///
    /// # Arguments
    /// * `d1` A 32-bit word
    /// * `d2` A 16-bit word
    /// * `d3` A 16-bit word
    /// * `d4` Array of 8 octets
    pub fn from_fields(d1: u32, d2: u16, d3: u16, d4: &[u8]) -> Uuid {
        use std::unstable::intrinsics::{to_be16, to_be32};

        // First construct a temporary field-based struct
        let mut fields = UuidFields {
                data1: 0,
                data2: 0,
                data3: 0,
                data4: [0, ..8]
        };

        fields.data1 = to_be32(d1 as i32) as u32;
        fields.data2 = to_be16(d2 as i16) as u16;
        fields.data3 = to_be16(d3 as i16) as u16;
        vec::bytes::copy_memory(fields.data4, d4, 8);

        unsafe {
            transmute(fields)
        }
    }

    /// Creates a UUID using the supplied bytes
    ///
    /// # Arguments
    /// * `b` An array or slice of 16 bytes
    pub fn from_bytes(b: &[u8]) -> Option<Uuid> {
        if b.len() != 16 {
            return None
        }

        let mut uuid = Uuid{ bytes: [0, .. 16] };
        unsafe {
            vec::raw::copy_memory(uuid.bytes, b, 16);
        }
        Some(uuid)
    }

    /// Specifies the variant of the UUID structure
    fn set_variant(&mut self, v: UuidVariant) {
        // Octet 8 contains the variant in the most significant 3 bits
        match v {
            VariantNCS =>        // b0xx...
                self.bytes[8] =  self.bytes[8] & 0x7f,
            VariantRFC4122 =>    // b10x...
                self.bytes[8] = (self.bytes[8] & 0x3f) | 0x80,
            VariantMicrosoft =>  // b110...
                self.bytes[8] = (self.bytes[8] & 0x1f) | 0xc0,
            VariantFuture =>     // b111...
                self.bytes[8] = (self.bytes[8] & 0x1f) | 0xe0,
        }
    }

    /// Returns the variant of the UUID structure
    ///
    /// This determines the interpretation of the structure of the UUID.
    /// Currently only the RFC4122 variant is generated by this module.
    ///
    /// * [Variant Reference](http://tools.ietf.org/html/rfc4122#section-4.1.1)
    pub fn get_variant(&self) -> Option<UuidVariant> {
        if self.bytes[8] & 0x80 == 0x00 {
            Some(VariantNCS)
        } else if self.bytes[8] & 0xc0 == 0x80 {
            Some(VariantRFC4122)
        } else if self.bytes[8] & 0xe0 == 0xc0  {
            Some(VariantMicrosoft)
        } else if self.bytes[8] & 0xe0 == 0xe0 {
            Some(VariantFuture)
        } else  {
            None
        }
    }

    /// Specifies the version number of the UUID
    fn set_version(&mut self, v: UuidVersion) {
        self.bytes[6] = (self.bytes[6] & 0xF) | ((v as u8) << 4);
    }

    /// Returns the version number of the UUID
    ///
    /// This represents the algorithm used to generate the contents.
    ///
    /// Currently only the Random (V4) algorithm is supported by this
    /// module.  There are security and privacy implications for using
    /// older versions - see [Wikipedia: Universally Unique Identifier](
    /// http://en.wikipedia.org/wiki/Universally_unique_identifier) for
    /// details.
    ///
    /// * [Version Reference](http://tools.ietf.org/html/rfc4122#section-4.1.3)
    pub fn get_version_num(&self) -> uint {
        (self.bytes[6] >> 4) as uint
    }

    /// Returns the version of the UUID
    ///
    /// This represents the algorithm used to generate the contents
    pub fn get_version(&self) -> Option<UuidVersion> {
        let v = (self.bytes[6] >> 4);
        match v {
            1 => Some(Version1Mac),
            2 => Some(Version2Dce),
            3 => Some(Version3Md5),
            4 => Some(Version4Random),
            5 => Some(Version5Sha1),
            _ => None
        }
    }

    /// Return an array of 16 octets containing the UUID data
    pub fn to_bytes<'a>(&'a self) -> &'a [u8] {
        self.bytes.as_slice()
    }

    /// Returns the UUID as a string of 16 hexadecimal digits
    ///
    /// Example: `936DA01F9ABD4d9d80C702AF85C822A8`
    pub fn to_simple_str(&self) -> ~str {
        let mut s: ~[u8] = vec::from_elem(32, 0u8);
        for i in range(0u, 16u) {
            let digit = format!("{:02x}", self.bytes[i] as uint);
            s[i*2+0] = digit[0];
            s[i*2+1] = digit[1];
        }
        str::from_utf8(s)
    }

    /// Returns a string of hexadecimal digits, separated into groups with a hypen
    ///
    /// Example: `550e8400-e29b-41d4-a716-446655440000`
    pub fn to_hyphenated_str(&self) -> ~str {
        use std::unstable::intrinsics::{to_be16, to_be32};
        // Convert to field-based struct as it matches groups in output.
        // Ensure fields are in network byte order, as per RFC.
        let mut uf: UuidFields;
        unsafe {
            uf = transmute_copy(&self.bytes);
        }
        uf.data1 = to_be32(uf.data1 as i32) as u32;
        uf.data2 = to_be16(uf.data2 as i16) as u16;
        uf.data3 = to_be16(uf.data3 as i16) as u16;
        let s = format!("{:08x}-{:04x}-{:04x}-{:02x}{:02x}-\
                         {:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            uf.data1,
            uf.data2, uf.data3,
            uf.data4[0], uf.data4[1],
            uf.data4[2], uf.data4[3], uf.data4[4],
            uf.data4[5], uf.data4[6], uf.data4[7]);
        s
    }

    /// Returns the UUID formatted as a full URN string
    ///
    /// This is the same as the hyphenated format, but with the "urn:uuid:" prefix.
    ///
    /// Example: `urn:uuid:F9168C5E-CEB2-4faa-B6BF-329BF39FA1E4`
    pub fn to_urn_str(&self) -> ~str {
        "urn:uuid:" + self.to_hyphenated_str()
    }

    /// Parses a UUID from a string of hexadecimal digits with optional hyphens
    ///
    /// Any of the formats generated by this module (simple, hyphenated, urn) are
    /// supported by this parsing function.
    pub fn parse_string(us: &str) -> Result<Uuid, ParseError> {

        let mut us = us.clone();
        let orig_len = us.len();

        // Ensure length is valid for any of the supported formats
        if orig_len != 32 && orig_len != 36 && orig_len != 45 {
            return Err(ErrorInvalidLength(orig_len));
        }

        // Strip off URN prefix if present
        if us.starts_with("urn:uuid:") {
            us = us.slice(9, orig_len);
        }

        // Make sure all chars are either hex digits or hyphen
        for (i, c) in us.iter().enumerate() {
            match c {
                '0'..'9' | 'A'..'F' | 'a'..'f' | '-' => {},
                _ => return Err(ErrorInvalidCharacter(c, i)),
            }
        }

        // Split string up by hyphens into groups
        let hex_groups: ~[&str] = us.split_str_iter("-").collect();

        // Get the length of each group
        let group_lens: ~[uint] = hex_groups.iter().map(|&v| v.len()).collect();

        // Ensure the group lengths are valid
        match group_lens.len() {
            // Single group, no hyphens
            1 => {
                if group_lens[0] != 32 {
                    return Err(ErrorInvalidLength(group_lens[0]));
                }
            },
            // Five groups, hyphens in between each
            5 => {
                // Ensure each group length matches the expected
                for (i, (&gl, &expected)) in
                    group_lens.iter().zip(UuidGroupLens.iter()).enumerate() {
                    if gl != expected {
                        return Err(ErrorInvalidGroupLength(i, gl, expected))
                    }
                }
            },
            _ => {
                return Err(ErrorInvalidGroups(group_lens.len()));
            }
        }

        // Normalise into one long hex string
        let vs = hex_groups.concat();

        // At this point, we know we have a valid hex string, without hyphens
        assert!(vs.len() == 32);
        assert!(vs.iter().all(|c| c.is_digit_radix(16)));

        // Allocate output UUID buffer
        let mut ub = [0u8, ..16];

        // Extract each hex digit from the string
        for i in range(0u, 16u) {
            ub[i] = FromStrRadix::from_str_radix(vs.slice(i*2, (i+1)*2), 16).unwrap();
        }

        Ok(Uuid::from_bytes(ub).unwrap())
    }
}

impl Default for Uuid {
    /// Returns the nil UUID, which is all zeroes
    fn default() -> Uuid {
        Uuid::new_nil()
    }
}

impl Zero for Uuid {
    /// Returns the nil UUID, which is all zeroes
    fn zero() -> Uuid {
        Uuid::new_nil()
    }

    /// Tests if the UUID is nil or all zeroes
    fn is_zero(&self) -> bool {
        return self.bytes.iter().all(|&b| b == 0);
    }
}

impl Clone for Uuid {
    /// Returns a copy of the UUID
    fn clone(&self) -> Uuid {
        let mut clone = Uuid{ bytes: [0, .. 16] };
        vec::bytes::copy_memory(clone.bytes, self.bytes, 16);
        clone
    }
}

impl FromStr for Uuid {
    /// Parse a hex string and interpret as a UUID
    ///
    /// Accepted formats are a sequence of 32 hexadecimal characters,
    /// with or without hypens (grouped as 8, 4, 4, 4, 12).
    fn from_str(us: &str) -> Option<Uuid> {
        let result = Uuid::parse_string(us);
        match result {
            Ok(u) => Some(u),
            Err(_) => None
        }
    }
}

/// Convert the UUID to a hexadecimal-based string representation
impl ToStr for Uuid {
    fn to_str(&self) -> ~str {
        self.to_simple_str()
    }
}

/// Test two UUIDs for equality
///
/// UUIDs are equal only when they are byte-for-byte identical
impl Eq for Uuid {
    fn eq(&self, other: &Uuid) -> bool {
        self.bytes == other.bytes
    }
}

/// Test two UUIDs for equality
///
/// UUIDs are equal only when they are byte-for-byte identical
impl TotalEq for Uuid {
    fn equals(&self, other: &Uuid) -> bool {
        self.bytes == other.bytes
    }
}

/// Generates a random instance of UUID (V4 conformant)
impl rand::Rand for Uuid {
    #[inline]
    fn rand<R: rand::Rng>(rng: &mut R) -> Uuid {
        let ub = rng.gen_vec(16);
        let mut uuid = Uuid{ bytes: [0, .. 16] };
        vec::bytes::copy_memory(uuid.bytes, ub, 16);
        uuid.set_variant(VariantRFC4122);
        uuid.set_version(Version4Random);
        uuid
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::str;
    use std::rand;
    use std::num::Zero;

    #[test]
    fn test_new_nil() {
        let nil = Uuid::new_nil();
        let nb = nil.to_bytes();

        assert!(nb.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_zero() {
        let uz: Uuid = Zero::zero();
        let nz = Uuid::new_v4();

        assert!(uz.is_zero());
        assert!(! nz.is_zero());
    }

    #[test]
    fn test_new() {
        // Supported
        let uuid1 = Uuid::new(Version4Random).unwrap();
        let s = uuid1.to_simple_str();

        assert!(s.len() == 32);
        assert!(uuid1.get_version().unwrap() == Version4Random);

        // Test unsupported versions
        assert!(Uuid::new(Version1Mac) == None);
        assert!(Uuid::new(Version2Dce) == None);
        assert!(Uuid::new(Version3Md5) == None);
        assert!(Uuid::new(Version5Sha1) == None);
    }

    #[test]
    fn test_new_v4() {
        let uuid1 = Uuid::new_v4();

        assert!(uuid1.get_version().unwrap() == Version4Random);
        assert!(uuid1.get_variant().unwrap() == VariantRFC4122);
    }

    #[test]
    fn test_get_version() {
        let uuid1 = Uuid::new_v4();

        assert!(uuid1.get_version().unwrap() == Version4Random);
        assert!(uuid1.get_version_num() == 4);
    }

    #[test]
    fn test_get_variant() {
        let uuid1 = Uuid::new_v4();
        let uuid2 = Uuid::parse_string("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let uuid3 = Uuid::parse_string("67e55044-10b1-426f-9247-bb680e5fe0c8").unwrap();
        let uuid4 = Uuid::parse_string("936DA01F9ABD4d9dC0C702AF85C822A8").unwrap();
        let uuid5 = Uuid::parse_string("F9168C5E-CEB2-4faa-D6BF-329BF39FA1E4").unwrap();
        let uuid6 = Uuid::parse_string("f81d4fae-7dec-11d0-7765-00a0c91e6bf6").unwrap();

        assert!(uuid1.get_variant().unwrap() == VariantRFC4122);
        assert!(uuid2.get_variant().unwrap() == VariantRFC4122);
        assert!(uuid3.get_variant().unwrap() == VariantRFC4122);
        assert!(uuid4.get_variant().unwrap() == VariantMicrosoft);
        assert!(uuid5.get_variant().unwrap() == VariantMicrosoft);
        assert!(uuid6.get_variant().unwrap() == VariantNCS);
    }

    #[test]
    fn test_parse_uuid_v4() {

        // Invalid
        assert!(Uuid::parse_string("").is_err());
        assert!(Uuid::parse_string("!").is_err());
        assert!(Uuid::parse_string("F9168C5E-CEB2-4faa-B6BF-329BF39FA1E45").is_err());
        assert!(Uuid::parse_string("F9168C5E-CEB2-4faa-BBF-329BF39FA1E4").is_err());
        assert!(Uuid::parse_string("F9168C5E-CEB2-4faa-BGBF-329BF39FA1E4").is_err());
        assert!(Uuid::parse_string("F9168C5E-CEB2-4faa-B6BFF329BF39FA1E4").is_err());
        assert!(Uuid::parse_string("F9168C5E-CEB2-4faa").is_err());
        assert!(Uuid::parse_string("F9168C5E-CEB2-4faaXB6BFF329BF39FA1E4").is_err());
        assert!(Uuid::parse_string("F9168C5E-CEB-24fa-eB6BFF32-BF39FA1E4").is_err());
        assert!(Uuid::parse_string("01020304-1112-2122-3132-41424344").is_err());
        assert!(Uuid::parse_string("67e5504410b1426f9247bb680e5fe0c").is_err());
        assert!(Uuid::parse_string("67e5504410b1426f9247bb680e5fe0c88").is_err());
        assert!(Uuid::parse_string("67e5504410b1426f9247bb680e5fe0cg8").is_err());
        assert!(Uuid::parse_string("67e5504410b1426%9247bb680e5fe0c8").is_err());

        // Valid
        assert!(Uuid::parse_string("00000000000000000000000000000000").is_ok());
        assert!(Uuid::parse_string("67e55044-10b1-426f-9247-bb680e5fe0c8").is_ok());
        assert!(Uuid::parse_string("67e55044-10b1-426f-9247-bb680e5fe0c8").is_ok());
        assert!(Uuid::parse_string("F9168C5E-CEB2-4faa-B6BF-329BF39FA1E4").is_ok());
        assert!(Uuid::parse_string("67e5504410b1426f9247bb680e5fe0c8").is_ok());
        assert!(Uuid::parse_string("01020304-1112-2122-3132-414243444546").is_ok());
        assert!(Uuid::parse_string("urn:uuid:67e55044-10b1-426f-9247-bb680e5fe0c8").is_ok());

        // Nil
        let nil = Uuid::new_nil();
        assert!(Uuid::parse_string("00000000000000000000000000000000").unwrap()  == nil);
        assert!(Uuid::parse_string("00000000-0000-0000-0000-000000000000").unwrap() == nil);

        // Round-trip
        let uuid_orig = Uuid::new_v4();
        let orig_str = uuid_orig.to_str();
        let uuid_out = Uuid::parse_string(orig_str).unwrap();
        assert!(uuid_orig == uuid_out);

        // Test error reporting
        let e = Uuid::parse_string("67e5504410b1426f9247bb680e5fe0c").unwrap_err();
        assert!(match(e){ ErrorInvalidLength(n) => n==31, _ => false });

        let e = Uuid::parse_string("67e550X410b1426f9247bb680e5fe0cd").unwrap_err();
        assert!(match(e){ ErrorInvalidCharacter(c, n) => c=='X' && n==6, _ => false });

        let e = Uuid::parse_string("67e550-4105b1426f9247bb680e5fe0c").unwrap_err();
        assert!(match(e){ ErrorInvalidGroups(n) => n==2, _ => false });

        let e = Uuid::parse_string("F9168C5E-CEB2-4faa-B6BF1-02BF39FA1E4").unwrap_err();
        assert!(match(e){ ErrorInvalidGroupLength(g, n, e) => g==3 && n==5 && e==4, _ => false });
    }

    #[test]
    fn test_to_simple_str() {
        let uuid1 = Uuid::new_v4();
        let s = uuid1.to_simple_str();

        assert!(s.len() == 32);
        assert!(s.iter().all(|c| c.is_digit_radix(16)));
    }

    #[test]
    fn test_to_str() {
        let uuid1 = Uuid::new_v4();
        let s = uuid1.to_str();

        assert!(s.len() == 32);
        assert!(s.iter().all(|c| c.is_digit_radix(16)));
    }

    #[test]
    fn test_to_hyphenated_str() {
        let uuid1 = Uuid::new_v4();
        let s = uuid1.to_hyphenated_str();

        assert!(s.len() == 36);
        assert!(s.iter().all(|c| c.is_digit_radix(16) || c == '-'));
    }

    #[test]
    fn test_to_urn_str() {
        let uuid1 = Uuid::new_v4();
        let ss = uuid1.to_urn_str();
        let s = ss.slice(9, ss.len());

        assert!(ss.starts_with("urn:uuid:"));
        assert!(s.len() == 36);
        assert!(s.iter().all(|c| c.is_digit_radix(16) || c == '-'));
    }

    #[test]
    fn test_to_str_matching() {
        let uuid1 = Uuid::new_v4();

        let hs = uuid1.to_hyphenated_str();
        let ss = uuid1.to_str();

        let hsn = str::from_chars(hs.iter().filter(|&c| c != '-').collect::<~[char]>());

        assert!(hsn == ss);
    }

    #[test]
    fn test_string_roundtrip() {
        let uuid = Uuid::new_v4();

        let hs = uuid.to_hyphenated_str();
        let uuid_hs = Uuid::parse_string(hs).unwrap();
        assert!(uuid_hs == uuid);

        let ss = uuid.to_str();
        let uuid_ss = Uuid::parse_string(ss).unwrap();
        assert!(uuid_ss == uuid);
    }

    #[test]
    fn test_compare() {
        let uuid1 = Uuid::new_v4();
        let uuid2 = Uuid::new_v4();

        assert!(uuid1 == uuid1);
        assert!(uuid2 == uuid2);
        assert!(uuid1 != uuid2);
        assert!(uuid2 != uuid1);
    }

    #[test]
    fn test_from_fields() {
        let d1: u32 = 0xa1a2a3a4;
        let d2: u16 = 0xb1b2;
        let d3: u16 = 0xc1c2;
        let d4: ~[u8] = ~[0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8];

        let u = Uuid::from_fields(d1, d2, d3, d4);

        let expected = ~"a1a2a3a4b1b2c1c2d1d2d3d4d5d6d7d8";
        let result = u.to_simple_str();
        assert!(result == expected);
    }

    #[test]
    fn test_from_bytes() {
        let b = ~[ 0xa1, 0xa2, 0xa3, 0xa4, 0xb1, 0xb2, 0xc1, 0xc2,
                   0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8 ];

        let u = Uuid::from_bytes(b).unwrap();
        let expected = ~"a1a2a3a4b1b2c1c2d1d2d3d4d5d6d7d8";

        assert!(u.to_simple_str() == expected);
    }

    #[test]
    fn test_to_bytes() {
        let u = Uuid::new_v4();
        let ub = u.to_bytes();

        assert!(ub.len() == 16);
        assert!(! ub.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_bytes_roundtrip() {
        let b_in: [u8, ..16] = [ 0xa1, 0xa2, 0xa3, 0xa4, 0xb1, 0xb2, 0xc1, 0xc2,
                                 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8 ];

        let u = Uuid::from_bytes(b_in.clone()).unwrap();

        let b_out = u.to_bytes();

        assert!(b_in == b_out);
    }

    #[test]
    fn test_operator_eq() {
        let u1 = Uuid::new_v4();
        let u2 = u1.clone();
        let u3 = Uuid::new_v4();

        assert!(u1 == u1);
        assert!(u1 == u2);
        assert!(u2 == u1);

        assert!(u1 != u3);
        assert!(u3 != u1);
        assert!(u2 != u3);
        assert!(u3 != u2);
    }

    #[test]
    fn test_rand_rand() {
        let mut rng = rand::rng();
        let u: ~Uuid = rand::Rand::rand(&mut rng);
        let ub = u.to_bytes();

        assert!(ub.len() == 16);
        assert!(! ub.iter().all(|&b| b == 0));
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use test::BenchHarness;

    #[bench]
    pub fn create_uuids(bh: &mut BenchHarness) {
        do bh.iter {
            Uuid::new_v4();
        }
    }

    #[bench]
    pub fn uuid_to_str(bh: &mut BenchHarness) {
        let u = Uuid::new_v4();
        do bh.iter {
            u.to_str();
        }
    }

    #[bench]
    pub fn parse_str(bh: &mut BenchHarness) {
        let s = "urn:uuid:F9168C5E-CEB2-4faa-B6BF-329BF39FA1E4";
        do bh.iter {
            Uuid::parse_string(s);
        }
    }
}
