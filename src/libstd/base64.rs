#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];
use io::Reader;

trait ToBase64 {
    fn to_base64() -> ~str;
}

impl &[u8]: ToBase64 {
    fn to_base64() -> ~str {
        let chars = str::chars(
          ~"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        );

        let len = self.len();
        let mut s = ~"";
        str::reserve(s, ((len + 3u) / 4u) * 3u);

        let mut i = 0u;

        while i < len - (len % 3u) {
            let n = (self[i] as uint) << 16u |
                    (self[i + 1u] as uint) << 8u |
                    (self[i + 2u] as uint);

            // This 24-bit number gets separated into four 6-bit numbers.
            str::push_char(s, chars[(n >> 18u) & 63u]);
            str::push_char(s, chars[(n >> 12u) & 63u]);
            str::push_char(s, chars[(n >> 6u) & 63u]);
            str::push_char(s, chars[n & 63u]);

            i += 3u;
        }

        // Heh, would be cool if we knew this was exhaustive
        // (the dream of bounded integer types)
        match len % 3 {
          0 => (),
          1 => {
            let n = (self[i] as uint) << 16u;
            str::push_char(s, chars[(n >> 18u) & 63u]);
            str::push_char(s, chars[(n >> 12u) & 63u]);
            str::push_char(s, '=');
            str::push_char(s, '=');
          }
          2 => {
            let n = (self[i] as uint) << 16u | (self[i + 1u] as uint) << 8u;
            str::push_char(s, chars[(n >> 18u) & 63u]);
            str::push_char(s, chars[(n >> 12u) & 63u]);
            str::push_char(s, chars[(n >> 6u) & 63u]);
            str::push_char(s, '=');
          }
          _ => fail ~"Algebra is broken, please alert the math police"
        }

        s
    }
}

impl &str: ToBase64 {
    fn to_base64() -> ~str {
        str::to_bytes(self).to_base64()
    }
}

trait FromBase64 {
    fn from_base64() -> ~[u8];
}

impl ~[u8]: FromBase64 {
    fn from_base64() -> ~[u8] {
        if self.len() % 4u != 0u { fail ~"invalid base64 length"; }

        let len = self.len();
        let mut padding = 0u;

        if len != 0u {
            if self[len - 1u] == '=' as u8 { padding += 1u; }
            if self[len - 2u] == '=' as u8 { padding += 1u; }
        }

        let mut r = ~[];
        vec::reserve(r, (len / 4u) * 3u - padding);

        let mut i = 0u;
        while i < len {
            let mut n = 0u;

            for iter::repeat(4u) {
                let ch = self[i] as char;
                n <<= 6u;

                if ch >= 'A' && ch <= 'Z' {
                    n |= (ch as uint) - 0x41u;
                } else if ch >= 'a' && ch <= 'z' {
                    n |= (ch as uint) - 0x47u;
                } else if ch >= '0' && ch <= '9' {
                    n |= (ch as uint) + 0x04u;
                } else if ch == '+' {
                    n |= 0x3Eu;
                } else if ch == '/' {
                    n |= 0x3Fu;
                } else if ch == '=' {
                    match len - i {
                      1u => {
                        vec::push(r, ((n >> 16u) & 0xFFu) as u8);
                        vec::push(r, ((n >> 8u ) & 0xFFu) as u8);
                        return copy r;
                      }
                      2u => {
                        vec::push(r, ((n >> 10u) & 0xFFu) as u8);
                        return copy r;
                      }
                      _ => fail ~"invalid base64 padding"
                    }
                } else {
                    fail ~"invalid base64 character";
                }

                i += 1u;
            };

            vec::push(r, ((n >> 16u) & 0xFFu) as u8);
            vec::push(r, ((n >> 8u ) & 0xFFu) as u8);
            vec::push(r, ((n       ) & 0xFFu) as u8);
        }

        r
    }
}

impl ~str: FromBase64 {
    fn from_base64() -> ~[u8] {
        str::to_bytes(self).from_base64()
    }
}

#[cfg(test)]
mod tests {
    #[legacy_exports];
    #[test]
    fn test_to_base64() {
        assert (~"").to_base64()       == ~"";
        assert (~"f").to_base64()      == ~"Zg==";
        assert (~"fo").to_base64()     == ~"Zm8=";
        assert (~"foo").to_base64()    == ~"Zm9v";
        assert (~"foob").to_base64()   == ~"Zm9vYg==";
        assert (~"fooba").to_base64()  == ~"Zm9vYmE=";
        assert (~"foobar").to_base64() == ~"Zm9vYmFy";
    }

    #[test]
    fn test_from_base64() {
        assert (~"").from_base64() == str::to_bytes(~"");
        assert (~"Zg==").from_base64() == str::to_bytes(~"f");
        assert (~"Zm8=").from_base64() == str::to_bytes(~"fo");
        assert (~"Zm9v").from_base64() == str::to_bytes(~"foo");
        assert (~"Zm9vYg==").from_base64() == str::to_bytes(~"foob");
        assert (~"Zm9vYmE=").from_base64() == str::to_bytes(~"fooba");
        assert (~"Zm9vYmFy").from_base64() == str::to_bytes(~"foobar");
    }
}
