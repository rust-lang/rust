// We're testing x86 target specific features
//@only-target: x86_64 i686
//@compile-flags: -C target-feature=+sse4.2

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::mem::transmute;

fn main() {
    assert!(is_x86_feature_detected!("sse4.2"));

    unsafe {
        test_sse42();
    }
}

#[target_feature(enable = "sse4.2")]
unsafe fn test_sse42() {
    // Mostly copied from library/stdarch/crates/core_arch/src/x86/sse42.rs

    test_crc();
    test_cmp();
    test_str();
}

#[target_feature(enable = "sse4.2")]
unsafe fn test_crc() {
    #[target_feature(enable = "sse4.2")]
    unsafe fn test_mm_crc32_u8() {
        let crc = 0x2aa1e72b;
        let v = 0x2a;
        let i = _mm_crc32_u8(crc, v);
        assert_eq!(i, 0xf24122e4);

        let crc = 0x61343ec4;
        let v = 0xef;
        let i = _mm_crc32_u8(crc, v);
        assert_eq!(i, 0xb95511db);

        let crc = 0xbadeafe;
        let v = 0xc0;
        let i = _mm_crc32_u8(crc, v);
        assert_eq!(i, 0x9c905b7c);
    }
    test_mm_crc32_u8();

    #[target_feature(enable = "sse4.2")]
    unsafe fn test_mm_crc32_u16() {
        let crc = 0x8ecec3b5;
        let v = 0x22b;
        let i = _mm_crc32_u16(crc, v);
        assert_eq!(i, 0x13bb2fb);

        let crc = 0x150bc664;
        let v = 0xa6c0;
        let i = _mm_crc32_u16(crc, v);
        assert_eq!(i, 0xab04fe4e);

        let crc = 0xbadeafe;
        let v = 0xc0fe;
        let i = _mm_crc32_u16(crc, v);
        assert_eq!(i, 0x4b5fad4b);
    }
    test_mm_crc32_u16();

    #[target_feature(enable = "sse4.2")]
    unsafe fn test_mm_crc32_u32() {
        let crc = 0xae2912c8;
        let v = 0x845fed;
        let i = _mm_crc32_u32(crc, v);
        assert_eq!(i, 0xffae2ed1);

        let crc = 0x1a198fe3;
        let v = 0x885585c2;
        let i = _mm_crc32_u32(crc, v);
        assert_eq!(i, 0x22443a7b);

        let crc = 0xbadeafe;
        let v = 0xc0febeef;
        let i = _mm_crc32_u32(crc, v);
        assert_eq!(i, 0xb309502f);
    }
    test_mm_crc32_u32();

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.2")]
    unsafe fn test_mm_crc32_u64() {
        let crc = 0x7819dccd3e824;
        let v = 0x2a22b845fed;
        let i = _mm_crc32_u64(crc, v);
        assert_eq!(i, 0xbb6cdc6c);

        let crc = 0x6dd960387fe13819;
        let v = 0x1a7ea8fb571746b0;
        let i = _mm_crc32_u64(crc, v);
        assert_eq!(i, 0x315b4f6);

        let crc = 0xbadeafe;
        let v = 0xc0febeefdadafefe;
        let i = _mm_crc32_u64(crc, v);
        assert_eq!(i, 0x5b44f54f);
    }
    #[cfg(not(target_arch = "x86_64"))]
    unsafe fn test_mm_crc32_u64() {}
    test_mm_crc32_u64();
}

#[target_feature(enable = "sse4.2")]
unsafe fn test_cmp() {
    let a = _mm_set_epi64x(0x2a, 0);
    let b = _mm_set1_epi64x(0x00);
    let i = _mm_cmpgt_epi64(a, b);
    assert_eq_m128i(i, _mm_set_epi64x(0xffffffffffffffffu64 as i64, 0x00));
}

#[target_feature(enable = "sse4.2")]
unsafe fn test_str() {
    #[target_feature(enable = "sse4.2")]
    unsafe fn str_to_m128i(s: &[u8]) -> __m128i {
        assert!(s.len() <= 16);
        let slice = &mut [0u8; 16];
        std::ptr::copy_nonoverlapping(s.as_ptr(), slice.as_mut_ptr(), s.len());
        _mm_loadu_si128(slice.as_ptr() as *const _)
    }

    // Test the `_mm_cmpistrm` intrinsic.
    #[target_feature(enable = "sse4.2")]
    unsafe fn test_mm_cmpistrm() {
        let a = str_to_m128i(b"Hello! Good-Bye!");
        let b = str_to_m128i(b"hello! good-bye!");
        let i = _mm_cmpistrm::<_SIDD_UNIT_MASK>(a, b);
        #[rustfmt::skip]
        let res = _mm_setr_epi8(
            0x00, !0, !0, !0, !0, !0, !0, 0x00,
            !0, !0, !0, !0, 0x00, !0, !0, !0,
        );
        assert_eq_m128i(i, res);
    }
    test_mm_cmpistrm();

    // Test the `_mm_cmpistri` intrinsic.
    #[target_feature(enable = "sse4.2")]
    unsafe fn test_mm_cmpistri() {
        let a = str_to_m128i(b"Hello");
        let b = str_to_m128i(b"   Hello        ");
        let i = _mm_cmpistri::<_SIDD_CMP_EQUAL_ORDERED>(a, b);
        assert_eq!(3, i);
    }
    test_mm_cmpistri();

    // Test the `_mm_cmpistrz` intrinsic.
    #[target_feature(enable = "sse4.2")]
    unsafe fn test_mm_cmpistrz() {
        let a = str_to_m128i(b"");
        let b = str_to_m128i(b"Hello");
        let i = _mm_cmpistrz::<_SIDD_CMP_EQUAL_ORDERED>(a, b);
        assert_eq!(1, i);
    }
    test_mm_cmpistrz();

    // Test the `_mm_cmpistrc` intrinsic.
    #[target_feature(enable = "sse4.2")]
    unsafe fn test_mm_cmpistrc() {
        let a = str_to_m128i(b"                ");
        let b = str_to_m128i(b"       !        ");
        let i = _mm_cmpistrc::<_SIDD_UNIT_MASK>(a, b);
        assert_eq!(1, i);
    }
    test_mm_cmpistrc();

    // Test the `_mm_cmpistrs` intrinsic.
    #[target_feature(enable = "sse4.2")]
    unsafe fn test_mm_cmpistrs() {
        let a = str_to_m128i(b"Hello");
        let b = str_to_m128i(b"");
        let i = _mm_cmpistrs::<_SIDD_CMP_EQUAL_ORDERED>(a, b);
        assert_eq!(1, i);
    }
    test_mm_cmpistrs();

    // Test the `_mm_cmpistro` intrinsic.
    #[target_feature(enable = "sse4.2")]
    unsafe fn test_mm_cmpistro() {
        #[rustfmt::skip]
        let a_bytes = _mm_setr_epi8(
            0x00, 0x47, 0x00, 0x65, 0x00, 0x6c, 0x00, 0x6c,
            0x00, 0x6f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        );
        #[rustfmt::skip]
        let b_bytes = _mm_setr_epi8(
            0x00, 0x48, 0x00, 0x65, 0x00, 0x6c, 0x00, 0x6c,
            0x00, 0x6f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        );
        let a = a_bytes;
        let b = b_bytes;
        let i = _mm_cmpistro::<{ _SIDD_UWORD_OPS | _SIDD_UNIT_MASK }>(a, b);
        assert_eq!(0, i);
    }
    test_mm_cmpistro();

    // Test the `_mm_cmpistra` intrinsic.
    #[target_feature(enable = "sse4.2")]
    unsafe fn test_mm_cmpistra() {
        let a = str_to_m128i(b"");
        let b = str_to_m128i(b"Hello!!!!!!!!!!!");
        let i = _mm_cmpistra::<_SIDD_UNIT_MASK>(a, b);
        assert_eq!(1, i);
    }
    test_mm_cmpistra();

    // Test the `_mm_cmpestrm` intrinsic.
    #[target_feature(enable = "sse4.2")]
    unsafe fn test_mm_cmpestrm() {
        let a = str_to_m128i(b"Hello!");
        let b = str_to_m128i(b"Hello.");
        let i = _mm_cmpestrm::<_SIDD_UNIT_MASK>(a, 5, b, 5);
        #[rustfmt::skip]
        let r = _mm_setr_epi8(
            !0, !0, !0, !0, !0, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        );
        assert_eq_m128i(i, r);
    }
    test_mm_cmpestrm();

    // Test the `_mm_cmpestri` intrinsic.
    #[target_feature(enable = "sse4.2")]
    unsafe fn test_mm_cmpestri() {
        let a = str_to_m128i(b"bar - garbage");
        let b = str_to_m128i(b"foobar");
        let i = _mm_cmpestri::<_SIDD_CMP_EQUAL_ORDERED>(a, 3, b, 6);
        assert_eq!(3, i);
    }
    test_mm_cmpestri();

    // Test the `_mm_cmpestrz` intrinsic.
    #[target_feature(enable = "sse4.2")]
    unsafe fn test_mm_cmpestrz() {
        let a = str_to_m128i(b"");
        let b = str_to_m128i(b"Hello");
        let i = _mm_cmpestrz::<_SIDD_CMP_EQUAL_ORDERED>(a, 16, b, 6);
        assert_eq!(1, i);
    }
    test_mm_cmpestrz();

    // Test the `_mm_cmpestrs` intrinsic.
    #[target_feature(enable = "sse4.2")]
    unsafe fn test_mm_cmpestrc() {
        let va = str_to_m128i(b"!!!!!!!!");
        let vb = str_to_m128i(b"        ");
        let i = _mm_cmpestrc::<_SIDD_UNIT_MASK>(va, 7, vb, 7);
        assert_eq!(0, i);
    }
    test_mm_cmpestrc();

    // Test the `_mm_cmpestrs` intrinsic.
    #[target_feature(enable = "sse4.2")]
    unsafe fn test_mm_cmpestrs() {
        #[rustfmt::skip]
        let a_bytes = _mm_setr_epi8(
            0x00, 0x48, 0x00, 0x65, 0x00, 0x6c, 0x00, 0x6c,
            0x00, 0x6f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        );
        let a = a_bytes;
        let b = _mm_set1_epi8(0x00);
        let i = _mm_cmpestrs::<_SIDD_UWORD_OPS>(a, 8, b, 0);
        assert_eq!(0, i);
    }
    test_mm_cmpestrs();

    // Test the `_mm_cmpestro` intrinsic.
    #[target_feature(enable = "sse4.2")]
    unsafe fn test_mm_cmpestro() {
        let a = str_to_m128i(b"Hello");
        let b = str_to_m128i(b"World");
        let i = _mm_cmpestro::<_SIDD_UBYTE_OPS>(a, 5, b, 5);
        assert_eq!(0, i);
    }
    test_mm_cmpestro();

    // Test the `_mm_cmpestra` intrinsic.
    #[target_feature(enable = "sse4.2")]
    unsafe fn test_mm_cmpestra() {
        let a = str_to_m128i(b"Cannot match a");
        let b = str_to_m128i(b"Null after 14");
        let i = _mm_cmpestra::<{ _SIDD_CMP_EQUAL_EACH | _SIDD_UNIT_MASK }>(a, 14, b, 16);
        assert_eq!(1, i);
    }
    test_mm_cmpestra();

    // Additional tests not inside the standard library.

    // Test the subset functionality of the intrinsic.
    unsafe fn test_subset() {
        let a = str_to_m128i(b"ABCDEFG");
        let b = str_to_m128i(b"ABC UVW XYZ EFG");

        let i = _mm_cmpistrm::<{ _SIDD_CMP_EQUAL_ANY | _SIDD_UNIT_MASK }>(a, b);
        #[rustfmt::skip]
        let res = _mm_setr_epi8(
            !0, !0, !0, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, !0, !0, !0, 0x00,
        );
        assert_eq_m128i(i, res);
    }
    test_subset();

    // Properly test index generation.
    unsafe fn test_index() {
        let a = str_to_m128i(b"Hello");
        let b = str_to_m128i(b"Hello Hello H");

        let i = _mm_cmpistri::<{ _SIDD_CMP_EQUAL_EACH | _SIDD_LEAST_SIGNIFICANT }>(a, b);
        assert_eq!(i, 0);

        let i = _mm_cmpistri::<{ _SIDD_CMP_EQUAL_EACH | _SIDD_MOST_SIGNIFICANT }>(a, b);
        assert_eq!(i, 15);

        let a = str_to_m128i(b"Hello");
        let b = str_to_m128i(b"                ");
        let i = _mm_cmpistri::<{ _SIDD_CMP_EQUAL_EACH | _SIDD_MOST_SIGNIFICANT }>(a, b);
        assert_eq!(i, 16);
    }
    test_index();

    // Properly test the substring functionality of the intrinsics.
    #[target_feature(enable = "sse4.2")]
    unsafe fn test_substring() {
        let a = str_to_m128i(b"Hello");
        let b = str_to_m128i(b"Hello Hello H");

        let i = _mm_cmpistrm::<{ _SIDD_CMP_EQUAL_ORDERED | _SIDD_UNIT_MASK }>(a, b);
        #[rustfmt::skip]
        let res = _mm_setr_epi8(
            !0, 0x00, 0x00, 0x00, 0x00, 0x00, !0, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        );
        assert_eq_m128i(i, res);
    }
    test_substring();

    // Test the range functionality of the intrinsics.
    // Will also test signed values and word-sized values.
    #[target_feature(enable = "sse4.2")]
    unsafe fn test_ranges() {
        let a = _mm_setr_epi16(0, 1, 7, 8, 0, 0, -100, 100);
        let b = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);

        let i =
            _mm_cmpestrm::<{ _SIDD_SWORD_OPS | _SIDD_CMP_RANGES | _SIDD_UNIT_MASK }>(a, 2, b, 8);
        let res = _mm_setr_epi16(!0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(i, res);

        let i =
            _mm_cmpestrm::<{ _SIDD_SWORD_OPS | _SIDD_CMP_RANGES | _SIDD_UNIT_MASK }>(a, 3, b, 8);
        let res = _mm_setr_epi16(!0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m128i(i, res);

        let i =
            _mm_cmpestrm::<{ _SIDD_SWORD_OPS | _SIDD_CMP_RANGES | _SIDD_UNIT_MASK }>(a, 4, b, 8);
        let res = _mm_setr_epi16(!0, 0, 0, 0, 0, 0, !0, !0);
        assert_eq_m128i(i, res);

        let i =
            _mm_cmpestrm::<{ _SIDD_SWORD_OPS | _SIDD_CMP_RANGES | _SIDD_UNIT_MASK }>(a, 6, b, 8);
        let res = _mm_setr_epi16(!0, 0, 0, 0, 0, 0, !0, !0);
        assert_eq_m128i(i, res);

        let i =
            _mm_cmpestrm::<{ _SIDD_SWORD_OPS | _SIDD_CMP_RANGES | _SIDD_UNIT_MASK }>(a, 8, b, 8);
        let res = _mm_setr_epi16(!0, !0, !0, !0, !0, !0, !0, !0);
        assert_eq_m128i(i, res);
    }
    test_ranges();

    // Confirm that the polarity bits work as indended.
    #[target_feature(enable = "sse4.2")]
    unsafe fn test_polarity() {
        let a = str_to_m128i(b"Hello!");
        let b = str_to_m128i(b"hello?");

        let i = _mm_cmpistrm::<
            { (_SIDD_MASKED_NEGATIVE_POLARITY ^ _SIDD_NEGATIVE_POLARITY) | _SIDD_UNIT_MASK },
        >(a, b);
        #[rustfmt::skip]
        let res = _mm_setr_epi8(
            0x00, !0, !0, !0, !0, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        );
        assert_eq_m128i(i, res);

        let i = _mm_cmpistrm::<{ _SIDD_MASKED_NEGATIVE_POLARITY | _SIDD_UNIT_MASK }>(a, b);
        #[rustfmt::skip]
        let res = _mm_setr_epi8(
            !0, 0x00, 0x00, 0x00, 0x00, !0, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        );
        assert_eq_m128i(i, res);

        let i = _mm_cmpistrm::<{ _SIDD_NEGATIVE_POLARITY | _SIDD_UNIT_MASK }>(a, b);
        #[rustfmt::skip]
        let res = _mm_setr_epi8(
            !0, 0x00, 0x00, 0x00, 0x00, !0, !0, !0,
            !0, !0, !0, !0, !0, !0, !0, !0,
        );
        assert_eq_m128i(i, res);
    }
    test_polarity();

    // Test the code path in which the intrinsic is supposed to
    // return a bit mask instead of a byte mask.
    #[target_feature(enable = "sse4.2")]
    unsafe fn test_bitmask() {
        let a = str_to_m128i(b"Hello! Good-Bye!");
        let b = str_to_m128i(b"hello! good-bye!");

        let i = _mm_cmpistrm::<0>(a, b);
        #[rustfmt::skip]
        let res = _mm_setr_epi32(0b11101111_01111110, 0, 0, 0);
        assert_eq_m128i(i, res);

        let i = _mm_cmpistrm::<_SIDD_MASKED_NEGATIVE_POLARITY>(a, b);
        #[rustfmt::skip]
        let res = _mm_setr_epi32(0b00010000_10000001, 0, 0, 0);
        assert_eq_m128i(i, res);
    }
    test_bitmask();
}

#[track_caller]
#[target_feature(enable = "sse2")]
pub unsafe fn assert_eq_m128i(a: __m128i, b: __m128i) {
    assert_eq!(transmute::<_, [u64; 2]>(a), transmute::<_, [u64; 2]>(b))
}
