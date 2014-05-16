// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Base64 binary-to-text encoding
use std::str;
use std::fmt;

/// Available encoding character sets
pub enum CharacterSet {
    /// The standard character set (uses `+` and `/`)
    Standard,
    /// The URL safe character set (uses `-` and `_`)
    UrlSafe
}

/// Contains configuration parameters for `to_base64`.
pub struct Config {
    /// Character set to use
    pub char_set: CharacterSet,
    /// True to pad output with `=` characters
    pub pad: bool,
    /// `Some(len)` to wrap lines at `len`, `None` to disable line wrapping
    pub line_length: Option<uint>
}

/// Configuration for RFC 4648 standard base64 encoding
pub static STANDARD: Config =
    Config {char_set: Standard, pad: true, line_length: None};

/// Configuration for RFC 4648 base64url encoding
pub static URL_SAFE: Config =
    Config {char_set: UrlSafe, pad: false, line_length: None};

/// Configuration for RFC 2045 MIME base64 encoding
pub static MIME: Config =
    Config {char_set: Standard, pad: true, line_length: Some(76)};

static STANDARD_CHARS: &'static[u8] = bytes!("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                             "abcdefghijklmnopqrstuvwxyz",
                                             "0123456789+/");

static URLSAFE_CHARS: &'static[u8] = bytes!("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                            "abcdefghijklmnopqrstuvwxyz",
                                            "0123456789-_");

/// A trait for converting a value to base64 encoding.
pub trait ToBase64 {
    /// Converts the value of `self` to a base64 value following the specified
    /// format configuration, returning the owned string.
    fn to_base64(&self, config: Config) -> ~str;
}

impl<'a> ToBase64 for &'a [u8] {
    /**
     * Turn a vector of `u8` bytes into a base64 string.
     *
     * # Example
     *
     * ```rust
     * extern crate serialize;
     * use serialize::base64::{ToBase64, STANDARD};
     *
     * fn main () {
     *     let str = [52,32].to_base64(STANDARD);
     *     println!("base 64 output: {}", str);
     * }
     * ```
     */
    fn to_base64(&self, config: Config) -> ~str {
        let bytes = match config.char_set {
            Standard => STANDARD_CHARS,
            UrlSafe => URLSAFE_CHARS
        };

        let mut v = Vec::new();
        let mut i = 0;
        let mut cur_length = 0;
        let len = self.len();
        while i < len - (len % 3) {
            match config.line_length {
                Some(line_length) =>
                    if cur_length >= line_length {
                        v.push('\r' as u8);
                        v.push('\n' as u8);
                        cur_length = 0;
                    },
                None => ()
            }

            let n = (self[i] as u32) << 16 |
                    (self[i + 1] as u32) << 8 |
                    (self[i + 2] as u32);

            // This 24-bit number gets separated into four 6-bit numbers.
            v.push(bytes[((n >> 18) & 63) as uint]);
            v.push(bytes[((n >> 12) & 63) as uint]);
            v.push(bytes[((n >> 6 ) & 63) as uint]);
            v.push(bytes[(n & 63) as uint]);

            cur_length += 4;
            i += 3;
        }

        if len % 3 != 0 {
            match config.line_length {
                Some(line_length) =>
                    if cur_length >= line_length {
                        v.push('\r' as u8);
                        v.push('\n' as u8);
                    },
                None => ()
            }
        }

        // Heh, would be cool if we knew this was exhaustive
        // (the dream of bounded integer types)
        match len % 3 {
            0 => (),
            1 => {
                let n = (self[i] as u32) << 16;
                v.push(bytes[((n >> 18) & 63) as uint]);
                v.push(bytes[((n >> 12) & 63) as uint]);
                if config.pad {
                    v.push('=' as u8);
                    v.push('=' as u8);
                }
            }
            2 => {
                let n = (self[i] as u32) << 16 |
                    (self[i + 1u] as u32) << 8;
                v.push(bytes[((n >> 18) & 63) as uint]);
                v.push(bytes[((n >> 12) & 63) as uint]);
                v.push(bytes[((n >> 6 ) & 63) as uint]);
                if config.pad {
                    v.push('=' as u8);
                }
            }
            _ => fail!("Algebra is broken, please alert the math police")
        }

        unsafe {
            str::raw::from_utf8(v.as_slice()).to_owned()
        }
    }
}

/// A trait for converting from base64 encoded values.
pub trait FromBase64 {
    /// Converts the value of `self`, interpreted as base64 encoded data, into
    /// an owned vector of bytes, returning the vector.
    fn from_base64(&self) -> Result<Vec<u8>, FromBase64Error>;
}

/// Errors that can occur when decoding a base64 encoded string
pub enum FromBase64Error {
    /// The input contained a character not part of the base64 format
    InvalidBase64Character(char, uint),
    /// The input had an invalid length
    InvalidBase64Length,
}

impl fmt::Show for FromBase64Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            InvalidBase64Character(ch, idx) =>
                write!(f, "Invalid character '{}' at position {}", ch, idx),
            InvalidBase64Length => write!(f, "Invalid length"),
        }
    }
}

impl<'a> FromBase64 for &'a str {
    /**
     * Convert any base64 encoded string (literal, `@`, `&`, or `~`)
     * to the byte values it encodes.
     *
     * You can use the `StrBuf::from_utf8` function in `std::strbuf` to turn a
     * `Vec<u8>` into a string with characters corresponding to those values.
     *
     * # Example
     *
     * This converts a string literal to base64 and back.
     *
     * ```rust
     * extern crate serialize;
     * use serialize::base64::{ToBase64, FromBase64, STANDARD};
     *
     * fn main () {
     *     let hello_str = bytes!("Hello, World").to_base64(STANDARD);
     *     println!("base64 output: {}", hello_str);
     *     let res = hello_str.from_base64();
     *     if res.is_ok() {
     *       let opt_bytes = StrBuf::from_utf8(res.unwrap());
     *       if opt_bytes.is_ok() {
     *         println!("decoded from base64: {}", opt_bytes.unwrap());
     *       }
     *     }
     * }
     * ```
     */
    fn from_base64(&self) -> Result<Vec<u8>, FromBase64Error> {
        let mut r = Vec::new();
        let mut buf: u32 = 0;
        let mut modulus = 0;

        let mut it = self.bytes().enumerate();
        for (idx, byte) in it {
            let val = byte as u32;

            match byte as char {
                'A'..'Z' => buf |= val - 0x41,
                'a'..'z' => buf |= val - 0x47,
                '0'..'9' => buf |= val + 0x04,
                '+'|'-' => buf |= 0x3E,
                '/'|'_' => buf |= 0x3F,
                '\r'|'\n' => continue,
                '=' => break,
                _ => return Err(InvalidBase64Character(self.char_at(idx), idx)),
            }

            buf <<= 6;
            modulus += 1;
            if modulus == 4 {
                modulus = 0;
                r.push((buf >> 22) as u8);
                r.push((buf >> 14) as u8);
                r.push((buf >> 6 ) as u8);
            }
        }

        for (idx, byte) in it {
            match byte as char {
                '='|'\r'|'\n' => continue,
                _ => return Err(InvalidBase64Character(self.char_at(idx), idx)),
            }
        }

        match modulus {
            2 => {
                r.push((buf >> 10) as u8);
            }
            3 => {
                r.push((buf >> 16) as u8);
                r.push((buf >> 8 ) as u8);
            }
            0 => (),
            _ => return Err(InvalidBase64Length),
        }

        Ok(r)
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    extern crate rand;
    use self::test::Bencher;
    use base64::{Config, FromBase64, ToBase64, STANDARD, URL_SAFE};

    #[test]
    fn test_to_base64_basic() {
        assert_eq!("".as_bytes().to_base64(STANDARD), "".to_owned());
        assert_eq!("f".as_bytes().to_base64(STANDARD), "Zg==".to_owned());
        assert_eq!("fo".as_bytes().to_base64(STANDARD), "Zm8=".to_owned());
        assert_eq!("foo".as_bytes().to_base64(STANDARD), "Zm9v".to_owned());
        assert_eq!("foob".as_bytes().to_base64(STANDARD), "Zm9vYg==".to_owned());
        assert_eq!("fooba".as_bytes().to_base64(STANDARD), "Zm9vYmE=".to_owned());
        assert_eq!("foobar".as_bytes().to_base64(STANDARD), "Zm9vYmFy".to_owned());
    }

    #[test]
    fn test_to_base64_line_break() {
        assert!(![0u8, ..1000].to_base64(Config {line_length: None, ..STANDARD})
                .contains("\r\n"));
        assert_eq!("foobar".as_bytes().to_base64(Config {line_length: Some(4),
                                                         ..STANDARD}),
                   "Zm9v\r\nYmFy".to_owned());
    }

    #[test]
    fn test_to_base64_padding() {
        assert_eq!("f".as_bytes().to_base64(Config {pad: false, ..STANDARD}), "Zg".to_owned());
        assert_eq!("fo".as_bytes().to_base64(Config {pad: false, ..STANDARD}), "Zm8".to_owned());
    }

    #[test]
    fn test_to_base64_url_safe() {
        assert_eq!([251, 255].to_base64(URL_SAFE), "-_8".to_owned());
        assert_eq!([251, 255].to_base64(STANDARD), "+/8=".to_owned());
    }

    #[test]
    fn test_from_base64_basic() {
        assert_eq!("".from_base64().unwrap().as_slice(), "".as_bytes());
        assert_eq!("Zg==".from_base64().unwrap().as_slice(), "f".as_bytes());
        assert_eq!("Zm8=".from_base64().unwrap().as_slice(), "fo".as_bytes());
        assert_eq!("Zm9v".from_base64().unwrap().as_slice(), "foo".as_bytes());
        assert_eq!("Zm9vYg==".from_base64().unwrap().as_slice(), "foob".as_bytes());
        assert_eq!("Zm9vYmE=".from_base64().unwrap().as_slice(), "fooba".as_bytes());
        assert_eq!("Zm9vYmFy".from_base64().unwrap().as_slice(), "foobar".as_bytes());
    }

    #[test]
    fn test_from_base64_newlines() {
        assert_eq!("Zm9v\r\nYmFy".from_base64().unwrap().as_slice(),
                   "foobar".as_bytes());
        assert_eq!("Zm9vYg==\r\n".from_base64().unwrap().as_slice(),
                   "foob".as_bytes());
    }

    #[test]
    fn test_from_base64_urlsafe() {
        assert_eq!("-_8".from_base64().unwrap(), "+/8=".from_base64().unwrap());
    }

    #[test]
    fn test_from_base64_invalid_char() {
        assert!("Zm$=".from_base64().is_err())
        assert!("Zg==$".from_base64().is_err());
    }

    #[test]
    fn test_from_base64_invalid_padding() {
        assert!("Z===".from_base64().is_err());
    }

    #[test]
    fn test_base64_random() {
        use self::rand::{task_rng, random, Rng};

        for _ in range(0, 1000) {
            let times = task_rng().gen_range(1u, 100);
            let v = Vec::from_fn(times, |_| random::<u8>());
            assert_eq!(v.as_slice().to_base64(STANDARD).from_base64().unwrap().as_slice(),
                       v.as_slice());
        }
    }

    #[bench]
    pub fn bench_to_base64(b: &mut Bencher) {
        let s = "イロハニホヘト チリヌルヲ ワカヨタレソ ツネナラム \
                 ウヰノオクヤマ ケフコエテ アサキユメミシ ヱヒモセスン";
        b.iter(|| {
            s.as_bytes().to_base64(STANDARD);
        });
        b.bytes = s.len() as u64;
    }

    #[bench]
    pub fn bench_from_base64(b: &mut Bencher) {
        let s = "イロハニホヘト チリヌルヲ ワカヨタレソ ツネナラム \
                 ウヰノオクヤマ ケフコエテ アサキユメミシ ヱヒモセスン";
        let sb = s.as_bytes().to_base64(STANDARD);
        b.iter(|| {
            sb.from_base64().unwrap();
        });
        b.bytes = sb.len() as u64;
    }

}
