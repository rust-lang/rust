// We're testing x86 target specific features
//@only-target: x86_64 i686
//@compile-flags: -C target-feature=+gfni,+avx512f

// The constants in the tests below are just bit patterns. They should not
// be interpreted as integers; signedness does not make sense for them, but
// __mXXXi happens to be defined in terms of signed integers.
#![allow(overflowing_literals)]

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::hint::black_box;
use std::mem::{size_of, transmute};

const IDENTITY_BYTE: i32 = 0;
const CONSTANT_BYTE: i32 = 0x63;

fn main() {
    // Mostly copied from library/stdarch/crates/core_arch/src/x86/gfni.rs

    assert!(is_x86_feature_detected!("avx512f"));
    assert!(is_x86_feature_detected!("gfni"));

    unsafe {
        let byte_mul_test_data = generate_byte_mul_test_data();
        let affine_mul_test_data_identity = generate_affine_mul_test_data(IDENTITY_BYTE as u8);
        let affine_mul_test_data_constant = generate_affine_mul_test_data(CONSTANT_BYTE as u8);
        let inv_tests_data = generate_inv_tests_data();

        test_mm512_gf2p8mul_epi8(&byte_mul_test_data);
        test_mm256_gf2p8mul_epi8(&byte_mul_test_data);
        test_mm_gf2p8mul_epi8(&byte_mul_test_data);
        test_mm512_gf2p8affine_epi64_epi8(&byte_mul_test_data, &affine_mul_test_data_identity);
        test_mm256_gf2p8affine_epi64_epi8(&byte_mul_test_data, &affine_mul_test_data_identity);
        test_mm_gf2p8affine_epi64_epi8(&byte_mul_test_data, &affine_mul_test_data_identity);
        test_mm512_gf2p8affineinv_epi64_epi8(&inv_tests_data, &affine_mul_test_data_constant);
        test_mm256_gf2p8affineinv_epi64_epi8(&inv_tests_data, &affine_mul_test_data_constant);
        test_mm_gf2p8affineinv_epi64_epi8(&inv_tests_data, &affine_mul_test_data_constant);
    }
}

#[target_feature(enable = "gfni,avx512f")]
unsafe fn test_mm512_gf2p8mul_epi8(
    byte_mul_test_data: &([u8; NUM_TEST_ENTRIES], [u8; NUM_TEST_ENTRIES], [u8; NUM_TEST_ENTRIES]),
) {
    let (left, right, expected) = byte_mul_test_data;

    for i in 0..NUM_TEST_WORDS_512 {
        let left = load_m512i_word(left, i);
        let right = load_m512i_word(right, i);
        let expected = load_m512i_word(expected, i);
        let result = _mm512_gf2p8mul_epi8(left, right);
        assert_eq_m512i(result, expected);
    }
}

#[target_feature(enable = "gfni,avx")]
unsafe fn test_mm256_gf2p8mul_epi8(
    byte_mul_test_data: &([u8; NUM_TEST_ENTRIES], [u8; NUM_TEST_ENTRIES], [u8; NUM_TEST_ENTRIES]),
) {
    let (left, right, expected) = byte_mul_test_data;

    for i in 0..NUM_TEST_WORDS_256 {
        let left = load_m256i_word(left, i);
        let right = load_m256i_word(right, i);
        let expected = load_m256i_word(expected, i);
        let result = _mm256_gf2p8mul_epi8(left, right);
        assert_eq_m256i(result, expected);
    }
}

#[target_feature(enable = "gfni")]
unsafe fn test_mm_gf2p8mul_epi8(
    byte_mul_test_data: &([u8; NUM_TEST_ENTRIES], [u8; NUM_TEST_ENTRIES], [u8; NUM_TEST_ENTRIES]),
) {
    let (left, right, expected) = byte_mul_test_data;

    for i in 0..NUM_TEST_WORDS_128 {
        let left = load_m128i_word(left, i);
        let right = load_m128i_word(right, i);
        let expected = load_m128i_word(expected, i);
        let result = _mm_gf2p8mul_epi8(left, right);
        assert_eq_m128i(result, expected);
    }
}

#[target_feature(enable = "gfni,avx512f")]
unsafe fn test_mm512_gf2p8affine_epi64_epi8(
    byte_mul_test_data: &([u8; NUM_TEST_ENTRIES], [u8; NUM_TEST_ENTRIES], [u8; NUM_TEST_ENTRIES]),
    affine_mul_test_data_identity: &(
        [u64; NUM_TEST_WORDS_64],
        [u8; NUM_TEST_ENTRIES],
        [u8; NUM_TEST_ENTRIES],
    ),
) {
    let identity: i64 = 0x01_02_04_08_10_20_40_80;
    let constant: i64 = 0;
    let identity = _mm512_set1_epi64(identity);
    let constant = _mm512_set1_epi64(constant);
    let constant_reference = _mm512_set1_epi8(CONSTANT_BYTE as i8);

    let (bytes, more_bytes, _) = byte_mul_test_data;
    let (matrices, vectors, references) = affine_mul_test_data_identity;

    for i in 0..NUM_TEST_WORDS_512 {
        let data = load_m512i_word(bytes, i);
        let result = _mm512_gf2p8affine_epi64_epi8::<IDENTITY_BYTE>(data, identity);
        assert_eq_m512i(result, data);
        let result = _mm512_gf2p8affine_epi64_epi8::<CONSTANT_BYTE>(data, constant);
        assert_eq_m512i(result, constant_reference);
        let data = load_m512i_word(more_bytes, i);
        let result = _mm512_gf2p8affine_epi64_epi8::<IDENTITY_BYTE>(data, identity);
        assert_eq_m512i(result, data);
        let result = _mm512_gf2p8affine_epi64_epi8::<CONSTANT_BYTE>(data, constant);
        assert_eq_m512i(result, constant_reference);

        let matrix = load_m512i_word(matrices, i);
        let vector = load_m512i_word(vectors, i);
        let reference = load_m512i_word(references, i);

        let result = _mm512_gf2p8affine_epi64_epi8::<IDENTITY_BYTE>(vector, matrix);
        assert_eq_m512i(result, reference);
    }
}

#[target_feature(enable = "gfni,avx")]
unsafe fn test_mm256_gf2p8affine_epi64_epi8(
    byte_mul_test_data: &([u8; NUM_TEST_ENTRIES], [u8; NUM_TEST_ENTRIES], [u8; NUM_TEST_ENTRIES]),
    affine_mul_test_data_identity: &(
        [u64; NUM_TEST_WORDS_64],
        [u8; NUM_TEST_ENTRIES],
        [u8; NUM_TEST_ENTRIES],
    ),
) {
    let identity: i64 = 0x01_02_04_08_10_20_40_80;
    let constant: i64 = 0;
    let identity = _mm256_set1_epi64x(identity);
    let constant = _mm256_set1_epi64x(constant);
    let constant_reference = _mm256_set1_epi8(CONSTANT_BYTE as i8);

    let (bytes, more_bytes, _) = byte_mul_test_data;
    let (matrices, vectors, references) = affine_mul_test_data_identity;

    for i in 0..NUM_TEST_WORDS_256 {
        let data = load_m256i_word(bytes, i);
        let result = _mm256_gf2p8affine_epi64_epi8::<IDENTITY_BYTE>(data, identity);
        assert_eq_m256i(result, data);
        let result = _mm256_gf2p8affine_epi64_epi8::<CONSTANT_BYTE>(data, constant);
        assert_eq_m256i(result, constant_reference);
        let data = load_m256i_word(more_bytes, i);
        let result = _mm256_gf2p8affine_epi64_epi8::<IDENTITY_BYTE>(data, identity);
        assert_eq_m256i(result, data);
        let result = _mm256_gf2p8affine_epi64_epi8::<CONSTANT_BYTE>(data, constant);
        assert_eq_m256i(result, constant_reference);

        let matrix = load_m256i_word(matrices, i);
        let vector = load_m256i_word(vectors, i);
        let reference = load_m256i_word(references, i);

        let result = _mm256_gf2p8affine_epi64_epi8::<IDENTITY_BYTE>(vector, matrix);
        assert_eq_m256i(result, reference);
    }
}

#[target_feature(enable = "gfni")]
unsafe fn test_mm_gf2p8affine_epi64_epi8(
    byte_mul_test_data: &([u8; NUM_TEST_ENTRIES], [u8; NUM_TEST_ENTRIES], [u8; NUM_TEST_ENTRIES]),
    affine_mul_test_data_identity: &(
        [u64; NUM_TEST_WORDS_64],
        [u8; NUM_TEST_ENTRIES],
        [u8; NUM_TEST_ENTRIES],
    ),
) {
    let identity: i64 = 0x01_02_04_08_10_20_40_80;
    let constant: i64 = 0;
    let identity = _mm_set1_epi64x(identity);
    let constant = _mm_set1_epi64x(constant);
    let constant_reference = _mm_set1_epi8(CONSTANT_BYTE as i8);

    let (bytes, more_bytes, _) = byte_mul_test_data;
    let (matrices, vectors, references) = affine_mul_test_data_identity;

    for i in 0..NUM_TEST_WORDS_128 {
        let data = load_m128i_word(bytes, i);
        let result = _mm_gf2p8affine_epi64_epi8::<IDENTITY_BYTE>(data, identity);
        assert_eq_m128i(result, data);
        let result = _mm_gf2p8affine_epi64_epi8::<CONSTANT_BYTE>(data, constant);
        assert_eq_m128i(result, constant_reference);
        let data = load_m128i_word(more_bytes, i);
        let result = _mm_gf2p8affine_epi64_epi8::<IDENTITY_BYTE>(data, identity);
        assert_eq_m128i(result, data);
        let result = _mm_gf2p8affine_epi64_epi8::<CONSTANT_BYTE>(data, constant);
        assert_eq_m128i(result, constant_reference);

        let matrix = load_m128i_word(matrices, i);
        let vector = load_m128i_word(vectors, i);
        let reference = load_m128i_word(references, i);

        let result = _mm_gf2p8affine_epi64_epi8::<IDENTITY_BYTE>(vector, matrix);
        assert_eq_m128i(result, reference);
    }
}

#[target_feature(enable = "gfni,avx512f")]
unsafe fn test_mm512_gf2p8affineinv_epi64_epi8(
    inv_tests_data: &([u8; NUM_BYTES], [u8; NUM_BYTES]),
    affine_mul_test_data_constant: &(
        [u64; NUM_TEST_WORDS_64],
        [u8; NUM_TEST_ENTRIES],
        [u8; NUM_TEST_ENTRIES],
    ),
) {
    let identity: i64 = 0x01_02_04_08_10_20_40_80;
    let identity = _mm512_set1_epi64(identity);

    // validate inversion
    let (inputs, results) = inv_tests_data;

    for i in 0..NUM_BYTES_WORDS_512 {
        let input = load_m512i_word(inputs, i);
        let reference = load_m512i_word(results, i);
        let result = _mm512_gf2p8affineinv_epi64_epi8::<IDENTITY_BYTE>(input, identity);
        let remultiplied = _mm512_gf2p8mul_epi8(result, input);
        assert_eq_m512i(remultiplied, reference);
    }

    // validate subsequent affine operation
    let (matrices, vectors, _affine_expected) = affine_mul_test_data_constant;

    for i in 0..NUM_TEST_WORDS_512 {
        let vector = load_m512i_word(vectors, i);
        let matrix = load_m512i_word(matrices, i);

        let inv_vec = _mm512_gf2p8affineinv_epi64_epi8::<IDENTITY_BYTE>(vector, identity);
        let reference = _mm512_gf2p8affine_epi64_epi8::<CONSTANT_BYTE>(inv_vec, matrix);
        let result = _mm512_gf2p8affineinv_epi64_epi8::<CONSTANT_BYTE>(vector, matrix);
        assert_eq_m512i(result, reference);
    }

    // validate everything by virtue of checking against the AES SBox
    const AES_S_BOX_MATRIX: i64 = 0xF1_E3_C7_8F_1F_3E_7C_F8;
    let sbox_matrix = _mm512_set1_epi64(AES_S_BOX_MATRIX);

    for i in 0..NUM_BYTES_WORDS_512 {
        let reference = load_m512i_word(&AES_S_BOX, i);
        let input = load_m512i_word(inputs, i);
        let result = _mm512_gf2p8affineinv_epi64_epi8::<CONSTANT_BYTE>(input, sbox_matrix);
        assert_eq_m512i(result, reference);
    }
}

#[target_feature(enable = "gfni,avx")]
unsafe fn test_mm256_gf2p8affineinv_epi64_epi8(
    inv_tests_data: &([u8; NUM_BYTES], [u8; NUM_BYTES]),
    affine_mul_test_data_constant: &(
        [u64; NUM_TEST_WORDS_64],
        [u8; NUM_TEST_ENTRIES],
        [u8; NUM_TEST_ENTRIES],
    ),
) {
    let identity: i64 = 0x01_02_04_08_10_20_40_80;
    let identity = _mm256_set1_epi64x(identity);

    // validate inversion
    let (inputs, results) = inv_tests_data;

    for i in 0..NUM_BYTES_WORDS_256 {
        let input = load_m256i_word(inputs, i);
        let reference = load_m256i_word(results, i);
        let result = _mm256_gf2p8affineinv_epi64_epi8::<IDENTITY_BYTE>(input, identity);
        let remultiplied = _mm256_gf2p8mul_epi8(result, input);
        assert_eq_m256i(remultiplied, reference);
    }

    // validate subsequent affine operation
    let (matrices, vectors, _affine_expected) = affine_mul_test_data_constant;

    for i in 0..NUM_TEST_WORDS_256 {
        let vector = load_m256i_word(vectors, i);
        let matrix = load_m256i_word(matrices, i);

        let inv_vec = _mm256_gf2p8affineinv_epi64_epi8::<IDENTITY_BYTE>(vector, identity);
        let reference = _mm256_gf2p8affine_epi64_epi8::<CONSTANT_BYTE>(inv_vec, matrix);
        let result = _mm256_gf2p8affineinv_epi64_epi8::<CONSTANT_BYTE>(vector, matrix);
        assert_eq_m256i(result, reference);
    }

    // validate everything by virtue of checking against the AES SBox
    const AES_S_BOX_MATRIX: i64 = 0xF1_E3_C7_8F_1F_3E_7C_F8;
    let sbox_matrix = _mm256_set1_epi64x(AES_S_BOX_MATRIX);

    for i in 0..NUM_BYTES_WORDS_256 {
        let reference = load_m256i_word(&AES_S_BOX, i);
        let input = load_m256i_word(inputs, i);
        let result = _mm256_gf2p8affineinv_epi64_epi8::<CONSTANT_BYTE>(input, sbox_matrix);
        assert_eq_m256i(result, reference);
    }
}

#[target_feature(enable = "gfni")]
unsafe fn test_mm_gf2p8affineinv_epi64_epi8(
    inv_tests_data: &([u8; NUM_BYTES], [u8; NUM_BYTES]),
    affine_mul_test_data_constant: &(
        [u64; NUM_TEST_WORDS_64],
        [u8; NUM_TEST_ENTRIES],
        [u8; NUM_TEST_ENTRIES],
    ),
) {
    let identity: i64 = 0x01_02_04_08_10_20_40_80;
    let identity = _mm_set1_epi64x(identity);

    // validate inversion
    let (inputs, results) = inv_tests_data;

    for i in 0..NUM_BYTES_WORDS_128 {
        let input = load_m128i_word(inputs, i);
        let reference = load_m128i_word(results, i);
        let result = _mm_gf2p8affineinv_epi64_epi8::<IDENTITY_BYTE>(input, identity);
        let remultiplied = _mm_gf2p8mul_epi8(result, input);
        assert_eq_m128i(remultiplied, reference);
    }

    // validate subsequent affine operation
    let (matrices, vectors, _affine_expected) = affine_mul_test_data_constant;

    for i in 0..NUM_TEST_WORDS_128 {
        let vector = load_m128i_word(vectors, i);
        let matrix = load_m128i_word(matrices, i);

        let inv_vec = _mm_gf2p8affineinv_epi64_epi8::<IDENTITY_BYTE>(vector, identity);
        let reference = _mm_gf2p8affine_epi64_epi8::<CONSTANT_BYTE>(inv_vec, matrix);
        let result = _mm_gf2p8affineinv_epi64_epi8::<CONSTANT_BYTE>(vector, matrix);
        assert_eq_m128i(result, reference);
    }

    // validate everything by virtue of checking against the AES SBox
    const AES_S_BOX_MATRIX: i64 = 0xF1_E3_C7_8F_1F_3E_7C_F8;
    let sbox_matrix = _mm_set1_epi64x(AES_S_BOX_MATRIX);

    for i in 0..NUM_BYTES_WORDS_128 {
        let reference = load_m128i_word(&AES_S_BOX, i);
        let input = load_m128i_word(inputs, i);
        let result = _mm_gf2p8affineinv_epi64_epi8::<CONSTANT_BYTE>(input, sbox_matrix);
        assert_eq_m128i(result, reference);
    }
}

/* Various utilities for processing SIMD values. */

#[target_feature(enable = "sse2")]
unsafe fn load_m128i_word<T>(data: &[T], word_index: usize) -> __m128i {
    let byte_offset = word_index * 16 / size_of::<T>();
    let pointer = data.as_ptr().add(byte_offset) as *const __m128i;
    _mm_loadu_si128(black_box(pointer))
}

#[target_feature(enable = "avx")]
unsafe fn load_m256i_word<T>(data: &[T], word_index: usize) -> __m256i {
    let byte_offset = word_index * 32 / size_of::<T>();
    let pointer = data.as_ptr().add(byte_offset) as *const __m256i;
    _mm256_loadu_si256(black_box(pointer))
}

#[target_feature(enable = "avx512f")]
unsafe fn load_m512i_word<T>(data: &[T], word_index: usize) -> __m512i {
    let byte_offset = word_index * 64 / size_of::<T>();
    let pointer = data.as_ptr().add(byte_offset) as *const __m512i;
    _mm512_loadu_si512(black_box(pointer))
}

#[track_caller]
#[target_feature(enable = "sse2")]
unsafe fn assert_eq_m128i(a: __m128i, b: __m128i) {
    assert_eq!(transmute::<_, [u64; 2]>(a), transmute::<_, [u64; 2]>(b))
}

#[track_caller]
#[target_feature(enable = "avx")]
unsafe fn assert_eq_m256i(a: __m256i, b: __m256i) {
    assert_eq!(transmute::<_, [u64; 4]>(a), transmute::<_, [u64; 4]>(b))
}

#[track_caller]
#[target_feature(enable = "avx512f")]
unsafe fn assert_eq_m512i(a: __m512i, b: __m512i) {
    assert_eq!(transmute::<_, [u64; 8]>(a), transmute::<_, [u64; 8]>(b))
}

/* Software implementation of the hardware intrinsics. */

fn mulbyte(left: u8, right: u8) -> u8 {
    // this implementation follows the description in
    // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_gf2p8mul_epi8
    const REDUCTION_POLYNOMIAL: u16 = 0x11b;
    let left: u16 = left.into();
    let right: u16 = right.into();
    let mut carryless_product: u16 = 0;

    // Carryless multiplication
    for i in 0..8 {
        if ((left >> i) & 0x01) != 0 {
            carryless_product ^= right << i;
        }
    }

    // reduction, adding in "0" where appropriate to clear out high bits
    // note that REDUCTION_POLYNOMIAL is zero in this context
    for i in (8..=14).rev() {
        if ((carryless_product >> i) & 0x01) != 0 {
            carryless_product ^= REDUCTION_POLYNOMIAL << (i - 8);
        }
    }

    carryless_product as u8
}

/// Calculates the bitwise XOR of all bits inside a byte.
fn parity(input: u8) -> u8 {
    let mut accumulator = 0;
    for i in 0..8 {
        accumulator ^= (input >> i) & 0x01;
    }
    accumulator
}

/// Calculates `matrix * x + b` inside the finite field GF(2).
fn mat_vec_multiply_affine(matrix: u64, x: u8, b: u8) -> u8 {
    // this implementation follows the description in
    // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_gf2p8affine_epi64_epi8
    let mut accumulator = 0;

    for bit in 0..8 {
        accumulator |= parity(x & matrix.to_le_bytes()[bit]) << (7 - bit);
    }

    accumulator ^ b
}

/* Test data generation. */

const NUM_TEST_WORDS_512: usize = 4;
const NUM_TEST_WORDS_256: usize = NUM_TEST_WORDS_512 * 2;
const NUM_TEST_WORDS_128: usize = NUM_TEST_WORDS_256 * 2;
const NUM_TEST_ENTRIES: usize = NUM_TEST_WORDS_512 * 64;
const NUM_TEST_WORDS_64: usize = NUM_TEST_WORDS_128 * 2;
const NUM_BYTES: usize = 256;
const NUM_BYTES_WORDS_128: usize = NUM_BYTES / 16;
const NUM_BYTES_WORDS_256: usize = NUM_BYTES_WORDS_128 / 2;
const NUM_BYTES_WORDS_512: usize = NUM_BYTES_WORDS_256 / 2;

fn generate_affine_mul_test_data(
    immediate: u8,
) -> ([u64; NUM_TEST_WORDS_64], [u8; NUM_TEST_ENTRIES], [u8; NUM_TEST_ENTRIES]) {
    let mut left: [u64; NUM_TEST_WORDS_64] = [0; NUM_TEST_WORDS_64];
    let mut right: [u8; NUM_TEST_ENTRIES] = [0; NUM_TEST_ENTRIES];
    let mut result: [u8; NUM_TEST_ENTRIES] = [0; NUM_TEST_ENTRIES];

    for i in 0..NUM_TEST_WORDS_64 {
        left[i] = (i as u64) * 103 * 101;
        for j in 0..8 {
            let j64 = j as u64;
            right[i * 8 + j] = ((left[i] + j64) % 256) as u8;
            result[i * 8 + j] = mat_vec_multiply_affine(left[i], right[i * 8 + j], immediate);
        }
    }

    (left, right, result)
}

fn generate_inv_tests_data() -> ([u8; NUM_BYTES], [u8; NUM_BYTES]) {
    let mut input: [u8; NUM_BYTES] = [0; NUM_BYTES];
    let mut result: [u8; NUM_BYTES] = [0; NUM_BYTES];

    for i in 0..NUM_BYTES {
        input[i] = (i % 256) as u8;
        result[i] = if i == 0 { 0 } else { 1 };
    }

    (input, result)
}

const AES_S_BOX: [u8; NUM_BYTES] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

fn generate_byte_mul_test_data()
-> ([u8; NUM_TEST_ENTRIES], [u8; NUM_TEST_ENTRIES], [u8; NUM_TEST_ENTRIES]) {
    let mut left: [u8; NUM_TEST_ENTRIES] = [0; NUM_TEST_ENTRIES];
    let mut right: [u8; NUM_TEST_ENTRIES] = [0; NUM_TEST_ENTRIES];
    let mut result: [u8; NUM_TEST_ENTRIES] = [0; NUM_TEST_ENTRIES];

    for i in 0..NUM_TEST_ENTRIES {
        left[i] = (i % 256) as u8;
        right[i] = left[i].wrapping_mul(101);
        result[i] = mulbyte(left[i], right[i]);
    }

    (left, right, result)
}
