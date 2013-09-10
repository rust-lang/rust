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

/// Available encoding character sets
pub enum CharacterSet {
    /// The standard character set (uses '+' and '/')
    Standard,
    /// The URL safe character set (uses '-' and '_')
    UrlSafe
}

/// Contains configuration parameters for to_base64
pub struct Config {
    /// Character set to use
    char_set: CharacterSet,
    /// True to pad output with '=' characters
    pad: bool,
    /// Some(len) to wrap lines at len, None to disable line wrapping
    line_length: Option<uint>
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

impl<'self> ToBase64 for &'self [u8] {
    /**
     * Turn a vector of `u8` bytes into a base64 string.
     *
     * # Example
     *
     * ~~~ {.rust}
     * extern mod extra;
     * use extra::base64::{ToBase64, standard};
     *
     * fn main () {
     *     let str = [52,32].to_base64(standard);
     *     printfln!("%s", str);
     * }
     * ~~~
     */
    fn to_base64(&self, config: Config) -> ~str {
        let bytes = match config.char_set {
            Standard => STANDARD_CHARS,
            UrlSafe => URLSAFE_CHARS
        };

        let mut v: ~[u8] = ~[];
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
            v.push(bytes[(n >> 18) & 63]);
            v.push(bytes[(n >> 12) & 63]);
            v.push(bytes[(n >> 6 ) & 63]);
            v.push(bytes[n & 63]);

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
                v.push(bytes[(n >> 18) & 63]);
                v.push(bytes[(n >> 12) & 63]);
                if config.pad {
                    v.push('=' as u8);
                    v.push('=' as u8);
                }
            }
            2 => {
                let n = (self[i] as u32) << 16 |
                    (self[i + 1u] as u32) << 8;
                v.push(bytes[(n >> 18) & 63]);
                v.push(bytes[(n >> 12) & 63]);
                v.push(bytes[(n >> 6 ) & 63]);
                if config.pad {
                    v.push('=' as u8);
                }
            }
            _ => fail!("Algebra is broken, please alert the math police")
        }

        unsafe {
            str::raw::from_utf8_owned(v)
        }
    }
}

/// A trait for converting from base64 encoded values.
pub trait FromBase64 {
    /// Converts the value of `self`, interpreted as base64 encoded data, into
    /// an owned vector of bytes, returning the vector.
    fn from_base64(&self) -> Result<~[u8], ~str>;
}

impl<'self> FromBase64 for &'self str {
    /**
     * Convert any base64 encoded string (literal, `@`, `&`, or `~`)
     * to the byte values it encodes.
     *
     * You can use the `from_utf8` function in `std::str`
     * to turn a `[u8]` into a string with characters corresponding to those
     * values.
     *
     * # Example
     *
     * This converts a string literal to base64 and back.
     *
     * ~~~ {.rust}
     * extern mod extra;
     * use extra::base64::{ToBase64, FromBase64, standard};
     * use std::str;
     *
     * fn main () {
     *     let hello_str = "Hello, World".to_base64(standard);
     *     printfln!("%s", hello_str);
     *     let bytes = hello_str.from_base64();
     *     printfln!("%?", bytes);
     *     let result_str = str::from_utf8(bytes);
     *     printfln!("%s", result_str);
     * }
     * ~~~
     */
    fn from_base64(&self) -> Result<~[u8], ~str> {
        let mut r = ~[];
        let mut buf: u32 = 0;
        let mut modulus = 0;

        let mut it = self.byte_iter().enumerate();
        for (idx, byte) in it {
            let val = byte as u32;

            match byte as char {
                'A'..'Z' => buf |= val - 0x41,
                'a'..'z' => buf |= val - 0x47,
                '0'..'9' => buf |= val + 0x04,
                '+'|'-' => buf |= 0x3E,
                '/'|'_' => buf |= 0x3F,
                '\r'|'\n' => loop,
                '=' => break,
                _ => return Err(fmt!("Invalid character '%c' at position %u",
                                     self.char_at(idx), idx))
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
            if (byte as char) != '=' {
                return Err(fmt!("Invalid character '%c' at position %u",
                                self.char_at(idx), idx));
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
            _ => return Err(~"Invalid Base64 length")
        }

        Ok(r)
    }
}

#[cfg(test)]
mod test {
    use test::BenchHarness;
    use base64::*;

    #[test]
    fn test_to_base64_basic() {
        assert_eq!("".as_bytes().to_base64(STANDARD), ~"");
        assert_eq!("f".as_bytes().to_base64(STANDARD), ~"Zg==");
        assert_eq!("fo".as_bytes().to_base64(STANDARD), ~"Zm8=");
        assert_eq!("foo".as_bytes().to_base64(STANDARD), ~"Zm9v");
        assert_eq!("foob".as_bytes().to_base64(STANDARD), ~"Zm9vYg==");
        assert_eq!("fooba".as_bytes().to_base64(STANDARD), ~"Zm9vYmE=");
        assert_eq!("foobar".as_bytes().to_base64(STANDARD), ~"Zm9vYmFy");
    }

    #[test]
    fn test_to_base64_line_break() {
        assert!(![0u8, 1000].to_base64(Config {line_length: None, ..STANDARD})
                .contains("\r\n"));
        assert_eq!("foobar".as_bytes().to_base64(Config {line_length: Some(4),
                                                         ..STANDARD}),
                   ~"Zm9v\r\nYmFy");
    }

    #[test]
    fn test_to_base64_padding() {
        assert_eq!("f".as_bytes().to_base64(Config {pad: false, ..STANDARD}), ~"Zg");
        assert_eq!("fo".as_bytes().to_base64(Config {pad: false, ..STANDARD}), ~"Zm8");
    }

    #[test]
    fn test_to_base64_url_safe() {
        assert_eq!([251, 255].to_base64(URL_SAFE), ~"-_8");
        assert_eq!([251, 255].to_base64(STANDARD), ~"+/8=");
    }

    #[test]
    fn test_from_base64_basic() {
        assert_eq!("".from_base64().unwrap(), "".as_bytes().to_owned());
        assert_eq!("Zg==".from_base64().unwrap(), "f".as_bytes().to_owned());
        assert_eq!("Zm8=".from_base64().unwrap(), "fo".as_bytes().to_owned());
        assert_eq!("Zm9v".from_base64().unwrap(), "foo".as_bytes().to_owned());
        assert_eq!("Zm9vYg==".from_base64().unwrap(), "foob".as_bytes().to_owned());
        assert_eq!("Zm9vYmE=".from_base64().unwrap(), "fooba".as_bytes().to_owned());
        assert_eq!("Zm9vYmFy".from_base64().unwrap(), "foobar".as_bytes().to_owned());
    }

    #[test]
    fn test_from_base64_newlines() {
        assert_eq!("Zm9v\r\nYmFy".from_base64().unwrap(),
                   "foobar".as_bytes().to_owned());
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
        use std::rand::{task_rng, random, RngUtil};
        use std::vec;

        do 1000.times {
            let times = task_rng().gen_uint_range(1, 100);
            let v = vec::from_fn(times, |_| random::<u8>());
            assert_eq!(v.to_base64(STANDARD).from_base64().unwrap(), v);
        }
    }

    #[bench]
    pub fn bench_to_base64(bh: & mut BenchHarness) {
        let s = "イロハニホヘト チリヌルヲ ワカヨタレソ ツネナラム \
                 ウヰノオクヤマ ケフコエテ アサキユメミシ ヱヒモセスン";
        do bh.iter {
            s.as_bytes().to_base64(STANDARD);
        }
        bh.bytes = s.len() as u64;
    }

    #[bench]
    pub fn bench_from_base64(bh: & mut BenchHarness) {
        let s = "イロハニホヘト チリヌルヲ ワカヨタレソ ツネナラム \
                 ウヰノオクヤマ ケフコエテ アサキユメミシ ヱヒモセスン";
        let b = s.as_bytes().to_base64(STANDARD);
        do bh.iter {
            b.from_base64();
        }
        bh.bytes = b.len() as u64;
    }

}
