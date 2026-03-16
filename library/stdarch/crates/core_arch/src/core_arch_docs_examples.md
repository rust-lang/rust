# Examples

First let's take a look at not actually using any intrinsics but instead
using LLVM's auto-vectorization to produce optimized vectorized code for
AVX2 and also for the default platform.

```rust
fn main() {
    let mut dst = [0];
    add_quickly(&[1], &[2], &mut dst);
    assert_eq!(dst[0], 3);
}

fn add_quickly(a: &[u8], b: &[u8], c: &mut [u8]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // Note that this `unsafe` block is safe because we're testing
        // that the `avx2` feature is indeed available on our CPU.
        if is_x86_feature_detected!("avx2") {
            return unsafe { add_quickly_avx2(a, b, c) };
        }
    }

    add_quickly_fallback(a, b, c)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn add_quickly_avx2(a: &[u8], b: &[u8], c: &mut [u8]) {
    add_quickly_fallback(a, b, c) // the function below is inlined here
}

fn add_quickly_fallback(a: &[u8], b: &[u8], c: &mut [u8]) {
    for ((a, b), c) in a.iter().zip(b).zip(c) {
        *c = *a + *b;
    }
}
```

Next up let's take a look at an example of manually using intrinsics. Here
we'll be using SSE4.1 features to implement hex encoding.

```
fn main() {
    let mut dst = [0; 32];
    hex_encode(b"\x01\x02\x03", &mut dst);
    assert_eq!(&dst[..6], b"010203");

    let mut src = [0; 16];
    for i in 0..16 {
        src[i] = (i + 1) as u8;
    }
    hex_encode(&src, &mut dst);
    assert_eq!(&dst, b"0102030405060708090a0b0c0d0e0f10");
}

pub fn hex_encode(src: &[u8], dst: &mut [u8]) {
    let len = src.len().checked_mul(2).unwrap();
    assert!(dst.len() >= len);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("sse4.1") {
            return unsafe { hex_encode_sse41(src, dst) };
        }
    }

    hex_encode_fallback(src, dst)
}

// translated from
// <https://github.com/Matherunner/bin2hex-sse/blob/master/base16_sse4.cpp>
#[target_feature(enable = "sse4.1")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn hex_encode_sse41(mut src: &[u8], dst: &mut [u8]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    unsafe {
        let ascii_zero = _mm_set1_epi8(b'0' as i8);
        let nines = _mm_set1_epi8(9);
        let ascii_a = _mm_set1_epi8((b'a' - 9 - 1) as i8);
        let and4bits = _mm_set1_epi8(0xf);

        let mut i = 0_isize;
        while src.len() >= 16 {
            let invec = _mm_loadu_si128(src.as_ptr() as *const _);

            let masked1 = _mm_and_si128(invec, and4bits);
            let masked2 = _mm_and_si128(_mm_srli_epi64(invec, 4), and4bits);

            // return 0xff corresponding to the elements > 9, or 0x00 otherwise
            let cmpmask1 = _mm_cmpgt_epi8(masked1, nines);
            let cmpmask2 = _mm_cmpgt_epi8(masked2, nines);

            // add '0' or the offset depending on the masks
            let masked1 = _mm_add_epi8(
                masked1,
                _mm_blendv_epi8(ascii_zero, ascii_a, cmpmask1),
            );
            let masked2 = _mm_add_epi8(
                masked2,
                _mm_blendv_epi8(ascii_zero, ascii_a, cmpmask2),
            );

            // interleave masked1 and masked2 bytes
            let res1 = _mm_unpacklo_epi8(masked2, masked1);
            let res2 = _mm_unpackhi_epi8(masked2, masked1);

            _mm_storeu_si128(dst.as_mut_ptr().offset(i * 2) as *mut _, res1);
            _mm_storeu_si128(
                dst.as_mut_ptr().offset(i * 2 + 16) as *mut _,
                res2,
            );
            src = &src[16..];
            i += 16;
        }

        let i = i as usize;
        hex_encode_fallback(src, &mut dst[i * 2..]);
    }
}

fn hex_encode_fallback(src: &[u8], dst: &mut [u8]) {
    fn hex(byte: u8) -> u8 {
        static TABLE: &[u8] = b"0123456789abcdef";
        TABLE[byte as usize]
    }

    for (byte, slots) in src.iter().zip(dst.chunks_mut(2)) {
        slots[0] = hex((*byte >> 4) & 0xf);
        slots[1] = hex(*byte & 0xf);
    }
}
```
