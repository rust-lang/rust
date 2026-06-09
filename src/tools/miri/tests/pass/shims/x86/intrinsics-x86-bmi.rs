// We're testing x86 target specific features
//@only-target: x86_64 i686
//@compile-flags: -C target-feature=+bmi1,+bmi2

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn main() {
    // BMI1 and BMI2 are independent from each other, so both must be checked.
    assert!(is_x86_feature_detected!("bmi1"));
    assert!(is_x86_feature_detected!("bmi2"));

    unsafe {
        test_bmi_32();
        test_bmi_64();
    }
}

/// Test the 32-bit variants of the intrinsics.
unsafe fn test_bmi_32() {
    unsafe fn test_bextr_u32() {
        let r = _bextr_u32(0b0101_0000u32, 4, 4);
        assert_eq!(r, 0b0000_0101u32);

        for i in 0..16 {
            assert_eq!(_bextr_u32(u32::MAX, i, 4), 0b1111);
            assert_eq!(_bextr_u32(u32::MAX, 4, i), (1 << i) - 1);
        }

        // Ensure that indices larger than the bit count are covered.
        // It is important to go above 32 in order to verify the bit selection
        // of the instruction.

        for i in 0..256 {
            // If the index is out of bounds, the original input won't be changed, thus the `min(32)`.
            assert_eq!(_bextr_u32(u32::MAX, 0, i).count_ones(), i.min(32));
        }

        for i in 0..256 {
            assert_eq!(_bextr_u32(u32::MAX, i, 0), 0);
        }

        // Test cases with completly random values. These cases also test
        // that the function works even if upper bits of the control value are set.
        assert_eq!(_bextr2_u32(0x7408a392, 0x54ef705), 0x3a0451c);
        assert_eq!(_bextr2_u32(0xbc5a3494, 0xdd193203), 0x178b4692);
        assert_eq!(_bextr2_u32(0xc0332325, 0xf96e207), 0x1806646);
    }
    test_bextr_u32();

    unsafe fn test_pext_u32() {
        let n = 0b1011_1110_1001_0011u32;

        let m0 = 0b0110_0011_1000_0101u32;
        let s0 = 0b0000_0000_0011_0101u32;

        let m1 = 0b1110_1011_1110_1111u32;
        let s1 = 0b0001_0111_0100_0011u32;

        // Testing of random values.
        assert_eq!(_pext_u32(n, m0), s0);
        assert_eq!(_pext_u32(n, m1), s1);
        assert_eq!(_pext_u32(0x12345678, 0xff00fff0), 0x00012567);

        // Testing of various identities.
        assert_eq!(_pext_u32(u32::MAX, u32::MAX), u32::MAX);
        assert_eq!(_pext_u32(u32::MAX, 0), 0);
        assert_eq!(_pext_u32(0, u32::MAX), 0);
    }
    test_pext_u32();

    unsafe fn test_pdep_u32() {
        let n = 0b1011_1110_1001_0011u32;

        let m0 = 0b0110_0011_1000_0101u32;
        let s0 = 0b0000_0010_0000_0101u32;

        let m1 = 0b1110_1011_1110_1111u32;
        let s1 = 0b1110_1001_0010_0011u32;

        // Testing of random values.
        assert_eq!(_pdep_u32(n, m0), s0);
        assert_eq!(_pdep_u32(n, m1), s1);
        assert_eq!(_pdep_u32(0x00012567, 0xff00fff0), 0x12005670);

        // Testing of various identities.
        assert_eq!(_pdep_u32(u32::MAX, u32::MAX), u32::MAX);
        assert_eq!(_pdep_u32(0, u32::MAX), 0);
        assert_eq!(_pdep_u32(u32::MAX, 0), 0);
    }
    test_pdep_u32();

    unsafe fn test_bzhi_u32() {
        let n = 0b1111_0010u32;
        let s = 0b0001_0010u32;
        assert_eq!(_bzhi_u32(n, 5), s);

        // Ensure that indices larger than the bit count are covered.
        // It is important to go above 32 in order to verify the bit selection
        // of the instruction.
        for i in 0..=512 {
            // The instruction only takes the lowest eight bits to generate the index, hence `i & 0xff`.
            // If the index is out of bounds, the original input won't be changed, thus the `min(32)`.
            let expected = 1u32.checked_shl((i & 0xff).min(32)).unwrap_or(0).wrapping_sub(1);
            let actual = _bzhi_u32(u32::MAX, i);
            assert_eq!(expected, actual);
        }
    }
    test_bzhi_u32();
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn test_bmi_64() {}

/// Test the 64-bit variants of the intrinsics.
#[cfg(target_arch = "x86_64")]
unsafe fn test_bmi_64() {
    unsafe fn test_bextr_u64() {
        let r = _bextr_u64(0b0101_0000u64, 4, 4);
        assert_eq!(r, 0b0000_0101u64);

        for i in 0..16 {
            assert_eq!(_bextr_u64(u64::MAX, i, 4), 0b1111);
            assert_eq!(_bextr_u64(u64::MAX, 32, i), (1 << i) - 1);
        }

        // Ensure that indices larger than the bit count are covered.
        // It is important to go above 64 in order to verify the bit selection
        // of the instruction.

        for i in 0..256 {
            // If the index is out of bounds, the original input won't be changed, thus the `min(64)`.
            assert_eq!(_bextr_u64(u64::MAX, 0, i).count_ones(), i.min(64));
        }

        for i in 0..256 {
            assert_eq!(_bextr_u64(u64::MAX, i, 0), 0);
        }

        // Test cases with completly random values. These cases also test
        // that the function works even if upper bits of the control value are set.
        assert_eq!(_bextr2_u64(0x4ff6cfbcea75f055, 0x216642e228425719), 0x27fb67de75);
        assert_eq!(_bextr2_u64(0xb05e991e6f6e1b6, 0xc76dd5d7f67dfc14), 0xb05e991e6f);
        assert_eq!(_bextr2_u64(0x5a3a629e323d848f, 0x95ac507d20e7719), 0x2d1d314f19);
    }
    test_bextr_u64();

    unsafe fn test_pext_u64() {
        let n = 0b1011_1110_1001_0011u64;

        let m0 = 0b0110_0011_1000_0101u64;
        let s0 = 0b0000_0000_0011_0101u64;

        let m1 = 0b1110_1011_1110_1111u64;
        let s1 = 0b0001_0111_0100_0011u64;

        // Testing of random values.
        assert_eq!(_pext_u64(n, m0), s0);
        assert_eq!(_pext_u64(n, m1), s1);
        assert_eq!(_pext_u64(0x12345678, 0xff00fff0), 0x00012567);

        // Testing of various identities.
        assert_eq!(_pext_u64(u64::MAX, u64::MAX), u64::MAX);
        assert_eq!(_pext_u64(u64::MAX, 0), 0);
        assert_eq!(_pext_u64(0, u64::MAX), 0);
    }
    test_pext_u64();

    unsafe fn test_pdep_u64() {
        let n = 0b1011_1110_1001_0011u64;

        let m0 = 0b0110_0011_1000_0101u64;
        let s0 = 0b0000_0010_0000_0101u64;

        let m1 = 0b1110_1011_1110_1111u64;
        let s1 = 0b1110_1001_0010_0011u64;

        // Testing of random values.
        assert_eq!(_pdep_u64(n, m0), s0);
        assert_eq!(_pdep_u64(n, m1), s1);
        assert_eq!(_pdep_u64(0x00012567, 0xff00fff0), 0x12005670);

        // Testing of various identities.
        assert_eq!(_pdep_u64(u64::MAX, u64::MAX), u64::MAX);
        assert_eq!(_pdep_u64(0, u64::MAX), 0);
        assert_eq!(_pdep_u64(u64::MAX, 0), 0);
    }
    test_pdep_u64();

    unsafe fn test_bzhi_u64() {
        let n = 0b1111_0010u64;
        let s = 0b0001_0010u64;
        assert_eq!(_bzhi_u64(n, 5), s);

        // Ensure that indices larger than the bit count are covered.
        // It is important to go above 255 in order to verify the bit selection
        // of the instruction.
        for i in 0..=512 {
            // The instruction only takes the lowest eight bits to generate the index, hence `i & 0xff`.
            // If the index is out of bounds, the original input won't be changed, thus the `min(64)`.
            let expected = 1u64.checked_shl((i & 0xff).min(64)).unwrap_or(0).wrapping_sub(1);
            let actual = _bzhi_u64(u64::MAX, i);
            assert_eq!(expected, actual);
        }
    }
    test_bzhi_u64();
}
