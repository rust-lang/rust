//! The [MD5][1] hash function.
//!
//! [1]: https://en.wikipedia.org/wiki/MD5

// The implementation is based on:
// http://people.csail.mit.edu/rivest/Md5.c

use std::convert::From;
use std::io::{Result, Write};
use std::mem;

/// A digest.
pub type Digest = [u8; 16];

/// A context.
#[derive(Copy)]
pub struct Context {
  handled: [u32; 2],
  buffer: [u32; 4],
  input: [u8; 64],
}

impl Clone for Context {
    #[inline]
    fn clone(&self) -> Context { *self }
}

const PADDING: [u8; 64] = [
    0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

impl Context {
    /// Create a context for computing a digest.
    #[inline]
    pub fn new() -> Context {
        Context {
            handled: [0, 0],
            buffer: [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476],
            input: unsafe { mem::uninitialized() },
        }
    }

    /// Consume data.
    pub fn consume(&mut self, data: &[u8]) {
        let mut input: [u32; 16] = unsafe { mem::uninitialized() };
        let mut k = ((self.handled[0] >> 3) & 0x3F) as usize;

        let length = data.len() as u32;
        if (self.handled[0] + (length << 3)) < self.handled[0] {
            self.handled[1] += 1;
        }
        self.handled[0] += length << 3;
        self.handled[1] += length >> 29;

        for &value in data {
            self.input[k] = value;
            k += 1;
            if k != 0x40 {
                continue;
            }
            let mut j = 0;
            for i in 0..16 {
                input[i] = ((self.input[j + 3] as u32) << 24) |
                           ((self.input[j + 2] as u32) << 16) |
                           ((self.input[j + 1] as u32) <<  8) |
                           ((self.input[j    ] as u32)      );
                j += 4;
            }
            transform(&mut self.buffer, &input);
            k = 0;
        }
    }

    /// Finalize and return the digest.
    pub fn compute(mut self) -> Digest {
        let mut input: [u32; 16] = unsafe { mem::uninitialized() };
        let k = ((self.handled[0] >> 3) & 0x3F) as usize;

        input[14] = self.handled[0];
        input[15] = self.handled[1];

        self.consume(&PADDING[..(if k < 56 { 56 - k } else { 120 - k })]);

        let mut j = 0;
        for i in 0..14 {
            input[i] = ((self.input[j + 3] as u32) << 24) |
                       ((self.input[j + 2] as u32) << 16) |
                       ((self.input[j + 1] as u32) <<  8) |
                       ((self.input[j    ] as u32)      );
            j += 4;
        }
        transform(&mut self.buffer, &input);

        let mut digest: Digest = unsafe { mem::uninitialized() };

        let mut j = 0;
        for i in 0..4 {
            digest[j    ] = ((self.buffer[i]      ) & 0xFF) as u8;
            digest[j + 1] = ((self.buffer[i] >>  8) & 0xFF) as u8;
            digest[j + 2] = ((self.buffer[i] >> 16) & 0xFF) as u8;
            digest[j + 3] = ((self.buffer[i] >> 24) & 0xFF) as u8;
            j += 4;
        }

        digest
    }
}

impl Write for Context {
    #[inline]
    fn write(&mut self, data: &[u8]) -> Result<usize> {
        self.consume(data);
        Ok(data.len())
    }

    #[inline]
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

impl From<Context> for Digest {
    #[inline]
    fn from(context: Context) -> Digest {
        context.compute()
    }
}

/// Compute the digest of data.
#[inline]
pub fn compute(data: &[u8]) -> Digest {
    let mut context = Context::new();
    context.consume(data);
    context.compute()
}

fn transform(buffer: &mut [u32; 4], input: &[u32; 16]) {
    let (mut a, mut b, mut c, mut d) = (buffer[0], buffer[1], buffer[2], buffer[3]);

    macro_rules! add(
        ($a:expr, $b:expr) => ($a.wrapping_add($b));
    );
    macro_rules! rotate(
        ($x:expr, $n:expr) => (($x << $n) | ($x >> (32 - $n)));
    );

    {
        macro_rules! F(
            ($x:expr, $y:expr, $z:expr) => (($x & $y) | (!$x & $z));
        );
        macro_rules! T(
            ($a:expr, $b:expr, $c:expr, $d:expr, $x:expr, $s:expr, $ac:expr) => ({
                $a = add!(add!(add!($a, F!($b, $c, $d)), $x), $ac);
                $a = rotate!($a, $s);
                $a = add!($a, $b);
            });
        );

        const S1: u32 =  7;
        const S2: u32 = 12;
        const S3: u32 = 17;
        const S4: u32 = 22;

        T!(a, b, c, d, input[ 0], S1, 3614090360);
        T!(d, a, b, c, input[ 1], S2, 3905402710);
        T!(c, d, a, b, input[ 2], S3,  606105819);
        T!(b, c, d, a, input[ 3], S4, 3250441966);
        T!(a, b, c, d, input[ 4], S1, 4118548399);
        T!(d, a, b, c, input[ 5], S2, 1200080426);
        T!(c, d, a, b, input[ 6], S3, 2821735955);
        T!(b, c, d, a, input[ 7], S4, 4249261313);
        T!(a, b, c, d, input[ 8], S1, 1770035416);
        T!(d, a, b, c, input[ 9], S2, 2336552879);
        T!(c, d, a, b, input[10], S3, 4294925233);
        T!(b, c, d, a, input[11], S4, 2304563134);
        T!(a, b, c, d, input[12], S1, 1804603682);
        T!(d, a, b, c, input[13], S2, 4254626195);
        T!(c, d, a, b, input[14], S3, 2792965006);
        T!(b, c, d, a, input[15], S4, 1236535329);
    }

    {
        macro_rules! F(
            ($x:expr, $y:expr, $z:expr) => (($x & $z) | ($y & !$z));
        );
        macro_rules! T(
            ($a:expr, $b:expr, $c:expr, $d:expr, $x:expr, $s:expr, $ac:expr) => ({
                $a = add!(add!(add!($a, F!($b, $c, $d)), $x), $ac);
                $a = rotate!($a, $s);
                $a = add!($a, $b);
            });
        );

        const S1: u32 =  5;
        const S2: u32 =  9;
        const S3: u32 = 14;
        const S4: u32 = 20;

        T!(a, b, c, d, input[ 1], S1, 4129170786);
        T!(d, a, b, c, input[ 6], S2, 3225465664);
        T!(c, d, a, b, input[11], S3,  643717713);
        T!(b, c, d, a, input[ 0], S4, 3921069994);
        T!(a, b, c, d, input[ 5], S1, 3593408605);
        T!(d, a, b, c, input[10], S2,   38016083);
        T!(c, d, a, b, input[15], S3, 3634488961);
        T!(b, c, d, a, input[ 4], S4, 3889429448);
        T!(a, b, c, d, input[ 9], S1,  568446438);
        T!(d, a, b, c, input[14], S2, 3275163606);
        T!(c, d, a, b, input[ 3], S3, 4107603335);
        T!(b, c, d, a, input[ 8], S4, 1163531501);
        T!(a, b, c, d, input[13], S1, 2850285829);
        T!(d, a, b, c, input[ 2], S2, 4243563512);
        T!(c, d, a, b, input[ 7], S3, 1735328473);
        T!(b, c, d, a, input[12], S4, 2368359562);
    }

    {
        macro_rules! F(
            ($x:expr, $y:expr, $z:expr) => ($x ^ $y ^ $z);
        );
        macro_rules! T(
            ($a:expr, $b:expr, $c:expr, $d:expr, $x:expr, $s:expr, $ac:expr) => ({
                $a = add!(add!(add!($a, F!($b, $c, $d)), $x), $ac);
                $a = rotate!($a, $s);
                $a = add!($a, $b);
            });
        );

        const S1: u32 =  4;
        const S2: u32 = 11;
        const S3: u32 = 16;
        const S4: u32 = 23;

        T!(a, b, c, d, input[ 5], S1, 4294588738);
        T!(d, a, b, c, input[ 8], S2, 2272392833);
        T!(c, d, a, b, input[11], S3, 1839030562);
        T!(b, c, d, a, input[14], S4, 4259657740);
        T!(a, b, c, d, input[ 1], S1, 2763975236);
        T!(d, a, b, c, input[ 4], S2, 1272893353);
        T!(c, d, a, b, input[ 7], S3, 4139469664);
        T!(b, c, d, a, input[10], S4, 3200236656);
        T!(a, b, c, d, input[13], S1,  681279174);
        T!(d, a, b, c, input[ 0], S2, 3936430074);
        T!(c, d, a, b, input[ 3], S3, 3572445317);
        T!(b, c, d, a, input[ 6], S4,   76029189);
        T!(a, b, c, d, input[ 9], S1, 3654602809);
        T!(d, a, b, c, input[12], S2, 3873151461);
        T!(c, d, a, b, input[15], S3,  530742520);
        T!(b, c, d, a, input[ 2], S4, 3299628645);
    }

    {
        macro_rules! F(
            ($x:expr, $y:expr, $z:expr) => ($y ^ ($x | !$z));
        );
        macro_rules! T(
            ($a:expr, $b:expr, $c:expr, $d:expr, $x:expr, $s:expr, $ac:expr) => ({
                $a = add!(add!(add!($a, F!($b, $c, $d)), $x), $ac);
                $a = rotate!($a, $s);
                $a = add!($a, $b);
            });
        );

        const S1: u32 =  6;
        const S2: u32 = 10;
        const S3: u32 = 15;
        const S4: u32 = 21;

        T!(a, b, c, d, input[ 0], S1, 4096336452);
        T!(d, a, b, c, input[ 7], S2, 1126891415);
        T!(c, d, a, b, input[14], S3, 2878612391);
        T!(b, c, d, a, input[ 5], S4, 4237533241);
        T!(a, b, c, d, input[12], S1, 1700485571);
        T!(d, a, b, c, input[ 3], S2, 2399980690);
        T!(c, d, a, b, input[10], S3, 4293915773);
        T!(b, c, d, a, input[ 1], S4, 2240044497);
        T!(a, b, c, d, input[ 8], S1, 1873313359);
        T!(d, a, b, c, input[15], S2, 4264355552);
        T!(c, d, a, b, input[ 6], S3, 2734768916);
        T!(b, c, d, a, input[13], S4, 1309151649);
        T!(a, b, c, d, input[ 4], S1, 4149444226);
        T!(d, a, b, c, input[11], S2, 3174756917);
        T!(c, d, a, b, input[ 2], S3,  718787259);
        T!(b, c, d, a, input[ 9], S4, 3951481745);
    }

    buffer[0] = add!(buffer[0], a);
    buffer[1] = add!(buffer[1], b);
    buffer[2] = add!(buffer[2], c);
    buffer[3] = add!(buffer[3], d);
}

#[cfg(test)]
mod tests {
    macro_rules! digest(
        ($string:expr) => ({
            let mut context = ::Context::new();
            context.consume($string.as_bytes());
            let mut digest = String::with_capacity(2 * 16);
            for x in &context.compute()[..] {
                digest.push_str(&format!("{:02x}", x));
            }
            digest
        });
    );

    #[test]
    fn compute() {
        let inputs = [
            "",
            "a",
            "abc",
            "message digest",
            "abcdefghijklmnopqrstuvwxyz",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            "12345678901234567890123456789012345678901234567890123456789012345678901234567890",
        ];

        let outputs = [
            "d41d8cd98f00b204e9800998ecf8427e",
            "0cc175b9c0f1b6a831c399e269772661",
            "900150983cd24fb0d6963f7d28e17f72",
            "f96b697d7cb7938d525a2f31aaf161d0",
            "c3fcd3d76192e4007dfb496cca67e13b",
            "d174ab98d277d9f5a5611c2c9f419d9f",
            "57edf4a22be3c955ac49da2e2107b67a",
        ];

        for (input, &output) in inputs.iter().zip(outputs.iter()) {
            assert_eq!(&digest!(input)[..], output);
        }
    }
}
