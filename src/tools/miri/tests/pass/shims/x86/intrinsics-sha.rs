// We're testing x86 target specific features
//@only-target: x86_64 i686
//@compile-flags: -C target-feature=+sha,+sse2,+ssse3,+sse4.1

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

macro_rules! rounds4 {
    ($abef:ident, $cdgh:ident, $rest:expr, $i:expr) => {{
        let k = K32X4[$i];
        let kv = _mm_set_epi32(k[0] as i32, k[1] as i32, k[2] as i32, k[3] as i32);
        let t1 = _mm_add_epi32($rest, kv);
        $cdgh = _mm_sha256rnds2_epu32($cdgh, $abef, t1);
        let t2 = _mm_shuffle_epi32(t1, 0x0E);
        $abef = _mm_sha256rnds2_epu32($abef, $cdgh, t2);
    }};
}

macro_rules! schedule_rounds4 {
    (
        $abef:ident, $cdgh:ident,
        $w0:expr, $w1:expr, $w2:expr, $w3:expr, $w4:expr,
        $i: expr
    ) => {{
        $w4 = schedule($w0, $w1, $w2, $w3);
        rounds4!($abef, $cdgh, $w4, $i);
    }};
}

fn main() {
    assert!(is_x86_feature_detected!("sha"));
    assert!(is_x86_feature_detected!("sse2"));
    assert!(is_x86_feature_detected!("ssse3"));
    assert!(is_x86_feature_detected!("sse4.1"));

    unsafe {
        test_sha256rnds2();
        test_sha256msg1();
        test_sha256msg2();
        test_sha256();
    }
}

#[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
unsafe fn test_sha256rnds2() {
    let test_vectors = [
        (
            [0x3c6ef372, 0xa54ff53a, 0x1f83d9ab, 0x5be0cd19],
            [0x6a09e667, 0xbb67ae85, 0x510e527f, 0x9b05688c],
            [0x592340c6, 0x17386142, 0x91a0b7b1, 0x94ffa30c],
            [0xeef39c6c, 0x4e7dfbc1, 0x467a98f3, 0xeb3d5616],
        ),
        (
            [0x6a09e667, 0xbb67ae85, 0x510e527f, 0x9b05688c],
            [0xeef39c6c, 0x4e7dfbc1, 0x467a98f3, 0xeb3d5616],
            [0x91a0b7b1, 0x94ffa30c, 0x592340c6, 0x17386142],
            [0x7e7f3c9d, 0x78db9a20, 0xd82fe6ed, 0xaf1f2704],
        ),
        (
            [0xeef39c6c, 0x4e7dfbc1, 0x467a98f3, 0xeb3d5616],
            [0x7e7f3c9d, 0x78db9a20, 0xd82fe6ed, 0xaf1f2704],
            [0x1a89c3f6, 0xf3b6e817, 0x7a5a8511, 0x8bcc35cf],
            [0xc9292f7e, 0x49137bd9, 0x7e5f9e08, 0xd10f9247],
        ),
    ];
    for (cdgh, abef, wk, expected) in test_vectors {
        let output_reg = _mm_sha256rnds2_epu32(set_arr(cdgh), set_arr(abef), set_arr(wk));
        let mut output = [0u32; 4];
        _mm_storeu_si128(output.as_mut_ptr().cast(), output_reg);
        // The values are stored as little endian, so we need to reverse them
        output.reverse();
        assert_eq!(output, expected);
    }
}

#[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
unsafe fn test_sha256msg1() {
    let test_vectors = [
        (
            [0x6f6d6521, 0x61776573, 0x20697320, 0x52757374],
            [0x6f6d6521, 0x61776573, 0x20697320, 0x52757374],
            [0x2da4b536, 0x77f29328, 0x541a4d59, 0x6afb680c],
        ),
        (
            [0x6f6d6521, 0x61776573, 0x20697320, 0x52757374],
            [0x6f6d6521, 0x61776573, 0x20697320, 0x52757374],
            [0x2da4b536, 0x77f29328, 0x541a4d59, 0x6afb680c],
        ),
        (
            [0x6f6d6521, 0x61776573, 0x20697320, 0x52757374],
            [0x6f6d6521, 0x61776573, 0x20697320, 0x52757374],
            [0x2da4b536, 0x77f29328, 0x541a4d59, 0x6afb680c],
        ),
    ];
    for (v0, v1, expected) in test_vectors {
        let output_reg = _mm_sha256msg1_epu32(set_arr(v0), set_arr(v1));
        let mut output = [0u32; 4];
        _mm_storeu_si128(output.as_mut_ptr().cast(), output_reg);
        // The values are stored as little endian, so we need to reverse them
        output.reverse();
        assert_eq!(output, expected);
    }
}

#[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
unsafe fn test_sha256msg2() {
    let test_vectors = [
        (
            [0x801a28aa, 0xe75ff849, 0xb591b2cc, 0x8b64db2c],
            [0x6f6d6521, 0x61776573, 0x20697320, 0x52757374],
            [0xe7c46c4e, 0x8ce92ccc, 0xd3c0f3ce, 0xe9745c78],
        ),
        (
            [0x171911ae, 0xe75ff849, 0xb591b2cc, 0x8b64db2c],
            [0xe7c46c4e, 0x8ce92ccc, 0xd3c0f3ce, 0xe9745c78],
            [0xc17c6ea3, 0xc4d10083, 0x712910cd, 0x3f41c8ce],
        ),
        (
            [0x6ce67e04, 0x5fb6ff76, 0xe1037a25, 0x3ebc5bda],
            [0xc17c6ea3, 0xc4d10083, 0x712910cd, 0x3f41c8ce],
            [0xf5ab4eff, 0x83d732a5, 0x9bb941af, 0xdf1d0a8c],
        ),
    ];
    for (v4, v3, expected) in test_vectors {
        let output_reg = _mm_sha256msg2_epu32(set_arr(v4), set_arr(v3));
        let mut output = [0u32; 4];
        _mm_storeu_si128(output.as_mut_ptr().cast(), output_reg);
        // The values are stored as little endian, so we need to reverse them
        output.reverse();
        assert_eq!(output, expected);
    }
}

#[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
unsafe fn set_arr(x: [u32; 4]) -> __m128i {
    _mm_set_epi32(x[0] as i32, x[1] as i32, x[2] as i32, x[3] as i32)
}

#[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
unsafe fn test_sha256() {
    use std::fmt::Write;

    /// The initial state of the hash engine.
    const INITIAL_STATE: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    // We don't want to bother with hash finalization algorithm so we just feed constant data.
    // This is the content that's being hashed - you can feed it to sha256sum and it'll output
    // the same hash (beware of newlines though).
    let first_block = *b"Rust is awesome!Rust is awesome!Rust is awesome!Rust is awesome!";
    // sha256 is fianlized by appending 0x80, then zeros and finally the data lenght at the
    // end.
    let mut final_block = [0; 64];
    final_block[0] = 0x80;
    final_block[(64 - 8)..].copy_from_slice(&(8u64 * 64).to_be_bytes());

    let mut state = INITIAL_STATE;
    digest_blocks(&mut state, &[first_block, final_block]);

    // We compare strings because it's easier to check the hex and the output of panic.
    let mut hash = String::new();
    for chunk in &state {
        write!(hash, "{:08x}", chunk).expect("writing to String doesn't fail");
    }
    assert_eq!(hash, "1b2293d21b17a0cb0c18737307c37333dea775eded18cefed45e50389f9f8184");
}

// Almost full SHA256 implementation copied from RustCrypto's sha2 crate
// https://github.com/RustCrypto/hashes/blob/6be8466247e936c415d8aafb848697f39894a386/sha2/src/sha256/x86.rs

#[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
unsafe fn schedule(v0: __m128i, v1: __m128i, v2: __m128i, v3: __m128i) -> __m128i {
    let t1 = _mm_sha256msg1_epu32(v0, v1);
    let t2 = _mm_alignr_epi8(v3, v2, 4);
    let t3 = _mm_add_epi32(t1, t2);
    _mm_sha256msg2_epu32(t3, v3)
}

// we use unaligned loads with `__m128i` pointers
#[expect(clippy::cast_ptr_alignment)]
#[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
unsafe fn digest_blocks(state: &mut [u32; 8], blocks: &[[u8; 64]]) {
    #[allow(non_snake_case)]
    let MASK: __m128i =
        _mm_set_epi64x(0x0C0D_0E0F_0809_0A0Bu64 as i64, 0x0405_0607_0001_0203u64 as i64);

    let state_ptr: *const __m128i = state.as_ptr().cast();
    let dcba = _mm_loadu_si128(state_ptr.add(0));
    let efgh = _mm_loadu_si128(state_ptr.add(1));

    let cdab = _mm_shuffle_epi32(dcba, 0xB1);
    let efgh = _mm_shuffle_epi32(efgh, 0x1B);
    let mut abef = _mm_alignr_epi8(cdab, efgh, 8);
    let mut cdgh = _mm_blend_epi16(efgh, cdab, 0xF0);

    for block in blocks {
        let abef_save = abef;
        let cdgh_save = cdgh;

        let block_ptr: *const __m128i = block.as_ptr().cast();
        let mut w0 = _mm_shuffle_epi8(_mm_loadu_si128(block_ptr.add(0)), MASK);
        let mut w1 = _mm_shuffle_epi8(_mm_loadu_si128(block_ptr.add(1)), MASK);
        let mut w2 = _mm_shuffle_epi8(_mm_loadu_si128(block_ptr.add(2)), MASK);
        let mut w3 = _mm_shuffle_epi8(_mm_loadu_si128(block_ptr.add(3)), MASK);
        let mut w4;

        rounds4!(abef, cdgh, w0, 0);
        rounds4!(abef, cdgh, w1, 1);
        rounds4!(abef, cdgh, w2, 2);
        rounds4!(abef, cdgh, w3, 3);
        schedule_rounds4!(abef, cdgh, w0, w1, w2, w3, w4, 4);
        schedule_rounds4!(abef, cdgh, w1, w2, w3, w4, w0, 5);
        schedule_rounds4!(abef, cdgh, w2, w3, w4, w0, w1, 6);
        schedule_rounds4!(abef, cdgh, w3, w4, w0, w1, w2, 7);
        schedule_rounds4!(abef, cdgh, w4, w0, w1, w2, w3, 8);
        schedule_rounds4!(abef, cdgh, w0, w1, w2, w3, w4, 9);
        schedule_rounds4!(abef, cdgh, w1, w2, w3, w4, w0, 10);
        schedule_rounds4!(abef, cdgh, w2, w3, w4, w0, w1, 11);
        schedule_rounds4!(abef, cdgh, w3, w4, w0, w1, w2, 12);
        schedule_rounds4!(abef, cdgh, w4, w0, w1, w2, w3, 13);
        schedule_rounds4!(abef, cdgh, w0, w1, w2, w3, w4, 14);
        schedule_rounds4!(abef, cdgh, w1, w2, w3, w4, w0, 15);

        abef = _mm_add_epi32(abef, abef_save);
        cdgh = _mm_add_epi32(cdgh, cdgh_save);
    }

    let feba = _mm_shuffle_epi32(abef, 0x1B);
    let dchg = _mm_shuffle_epi32(cdgh, 0xB1);
    let dcba = _mm_blend_epi16(feba, dchg, 0xF0);
    let hgef = _mm_alignr_epi8(dchg, feba, 8);

    let state_ptr_mut: *mut __m128i = state.as_mut_ptr().cast();
    _mm_storeu_si128(state_ptr_mut.add(0), dcba);
    _mm_storeu_si128(state_ptr_mut.add(1), hgef);
}

/// Swapped round constants for SHA-256 family of digests
pub static K32X4: [[u32; 4]; 16] = {
    let mut res = [[0u32; 4]; 16];
    let mut i = 0;
    while i < 16 {
        res[i] = [K32[4 * i + 3], K32[4 * i + 2], K32[4 * i + 1], K32[4 * i]];
        i += 1;
    }
    res
};

/// Round constants for SHA-256 family of digests
pub static K32: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];
