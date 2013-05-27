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

use core::prelude::*;

pub trait ToBase64 {
    fn to_base64(&self) -> ~str;
}

static CHARS: [char, ..64] = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'
];

impl<'self> ToBase64 for &'self [u8] {
    /**
     * Turn a vector of `u8` bytes into a base64 string.
     *
     * # Example
     *
     * ~~~ {.rust}
     * extern mod std;
     * use std::base64::ToBase64;
     *
     * fn main () {
     *     let str = [52,32].to_base64();
     *     println(fmt!("%s", str));
     * }
     * ~~~
     */
    fn to_base64(&self) -> ~str {
        let mut s = ~"";
        let len = self.len();
        str::reserve(&mut s, ((len + 3u) / 4u) * 3u);

        let mut i = 0u;

        while i < len - (len % 3u) {
            let n = (self[i] as uint) << 16u |
                    (self[i + 1u] as uint) << 8u |
                    (self[i + 2u] as uint);

            // This 24-bit number gets separated into four 6-bit numbers.
            str::push_char(&mut s, CHARS[(n >> 18u) & 63u]);
            str::push_char(&mut s, CHARS[(n >> 12u) & 63u]);
            str::push_char(&mut s, CHARS[(n >> 6u) & 63u]);
            str::push_char(&mut s, CHARS[n & 63u]);

            i += 3u;
        }

        // Heh, would be cool if we knew this was exhaustive
        // (the dream of bounded integer types)
        match len % 3 {
          0 => (),
          1 => {
            let n = (self[i] as uint) << 16u;
            str::push_char(&mut s, CHARS[(n >> 18u) & 63u]);
            str::push_char(&mut s, CHARS[(n >> 12u) & 63u]);
            str::push_char(&mut s, '=');
            str::push_char(&mut s, '=');
          }
          2 => {
            let n = (self[i] as uint) << 16u |
                (self[i + 1u] as uint) << 8u;
            str::push_char(&mut s, CHARS[(n >> 18u) & 63u]);
            str::push_char(&mut s, CHARS[(n >> 12u) & 63u]);
            str::push_char(&mut s, CHARS[(n >> 6u) & 63u]);
            str::push_char(&mut s, '=');
          }
          _ => fail!("Algebra is broken, please alert the math police")
        }
        s
    }
}

impl<'self> ToBase64 for &'self str {
    /**
     * Convert any string (literal, `@`, `&`, or `~`) to base64 encoding.
     *
     *
     * # Example
     *
     * ~~~ {.rust}
     * extern mod std;
     * use std::base64::ToBase64;
     *
     * fn main () {
     *     let str = "Hello, World".to_base64();
     *     println(fmt!("%s",str));
     * }
     * ~~~
     *
     */
    fn to_base64(&self) -> ~str {
        str::to_bytes(*self).to_base64()
    }
}

pub trait FromBase64 {
    fn from_base64(&self) -> ~[u8];
}

impl FromBase64 for ~[u8] {
    /**
     * Convert base64 `u8` vector into u8 byte values.
     * Every 4 encoded characters is converted into 3 octets, modulo padding.
     *
     * # Example
     *
     * ~~~ {.rust}
     * extern mod std;
     * use std::base64::ToBase64;
     * use std::base64::FromBase64;
     *
     * fn main () {
     *     let str = [52,32].to_base64();
     *     println(fmt!("%s", str));
     *     let bytes = str.from_base64();
     *     println(fmt!("%?",bytes));
     * }
     * ~~~
     */
    fn from_base64(&self) -> ~[u8] {
        if self.len() % 4u != 0u { fail!("invalid base64 length"); }

        let len = self.len();
        let mut padding = 0u;

        if len != 0u {
            if self[len - 1u] == '=' as u8 { padding += 1u; }
            if self[len - 2u] == '=' as u8 { padding += 1u; }
        }

        let mut r = vec::with_capacity((len / 4u) * 3u - padding);

        let mut i = 0u;
        while i < len {
            let mut n = 0u;

            for 4u.times {
                let ch = self[i] as char;
                n <<= 6u;

                match ch {
                    'A'..'Z' => n |= (ch as uint) - 0x41,
                    'a'..'z' => n |= (ch as uint) - 0x47,
                    '0'..'9' => n |= (ch as uint) + 0x04,
                    '+'      => n |= 0x3E,
                    '/'      => n |= 0x3F,
                    '='      => {
                        match len - i {
                            1u => {
                                r.push(((n >> 16u) & 0xFFu) as u8);
                                r.push(((n >> 8u ) & 0xFFu) as u8);
                                return copy r;
                            }
                            2u => {
                                r.push(((n >> 10u) & 0xFFu) as u8);
                                return copy r;
                            }
                            _ => fail!("invalid base64 padding")
                        }
                    }
                    _ => fail!("invalid base64 character")
                }

                i += 1u;
            };

            r.push(((n >> 16u) & 0xFFu) as u8);
            r.push(((n >> 8u ) & 0xFFu) as u8);
            r.push(((n       ) & 0xFFu) as u8);
        }
        r
    }
}

impl FromBase64 for ~str {
    /**
     * Convert any base64 encoded string (literal, `@`, `&`, or `~`)
     * to the byte values it encodes.
     *
     * You can use the `from_bytes` function in `core::str`
     * to turn a `[u8]` into a string with characters corresponding to those values.
     *
     * # Example
     *
     * This converts a string literal to base64 and back.
     *
     * ~~~ {.rust}
     * extern mod std;
     * use std::base64::ToBase64;
     * use std::base64::FromBase64;
     * use core::str;
     *
     * fn main () {
     *     let hello_str = "Hello, World".to_base64();
     *     println(fmt!("%s",hello_str));
     *     let bytes = hello_str.from_base64();
     *     println(fmt!("%?",bytes));
     *     let result_str = str::from_bytes(bytes);
     *     println(fmt!("%s",result_str));
     * }
     * ~~~
     */
    fn from_base64(&self) -> ~[u8] {
        str::to_bytes(*self).from_base64()
    }
}

#[cfg(test)]
mod tests {
    use core::str;

    #[test]
    fn test_to_base64() {
        assert_eq!((~"").to_base64(), ~"");
        assert!((~"f").to_base64()      == ~"Zg==");
        assert_eq!((~"fo").to_base64(), ~"Zm8=");
        assert_eq!((~"foo").to_base64(), ~"Zm9v");
        assert!((~"foob").to_base64()   == ~"Zm9vYg==");
        assert_eq!((~"fooba").to_base64(), ~"Zm9vYmE=");
        assert_eq!((~"foobar").to_base64(), ~"Zm9vYmFy");
    }

    #[test]
    fn test_from_base64() {
        assert_eq!((~"").from_base64(), str::to_bytes(""));
        assert!((~"Zg==").from_base64() == str::to_bytes("f"));
        assert_eq!((~"Zm8=").from_base64(), str::to_bytes("fo"));
        assert_eq!((~"Zm9v").from_base64(), str::to_bytes("foo"));
        assert!((~"Zm9vYg==").from_base64() == str::to_bytes("foob"));
        assert_eq!((~"Zm9vYmE=").from_base64(), str::to_bytes("fooba"))
        assert_eq!((~"Zm9vYmFy").from_base64(), str::to_bytes("foobar"));
    }
}
