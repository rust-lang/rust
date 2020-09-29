use super::*;

// The tests below compare the computed hashes to particular expected values
// in order to test that we produce the same results on different platforms,
// regardless of endianness and `usize` and `isize` size differences (this
// of course assumes we run these tests on platforms that differ in those
// ways). The expected values depend on the hashing algorithm used, so they
// need to be updated whenever StableHasher changes its hashing algorithm.

#[test]
fn test_hash_integers() {
    // Test that integers are handled consistently across platforms.
    let test_u8 = 0xAB_u8;
    let test_u16 = 0xFFEE_u16;
    let test_u32 = 0x445577AA_u32;
    let test_u64 = 0x01234567_13243546_u64;
    let test_u128 = 0x22114433_66557788_99AACCBB_EEDDFF77_u128;
    let test_usize = 0xD0C0B0A0_usize;

    let test_i8 = -100_i8;
    let test_i16 = -200_i16;
    let test_i32 = -300_i32;
    let test_i64 = -400_i64;
    let test_i128 = -500_i128;
    let test_isize = -600_isize;

    let mut h = StableHasher::new();
    test_u8.hash(&mut h);
    test_u16.hash(&mut h);
    test_u32.hash(&mut h);
    test_u64.hash(&mut h);
    test_u128.hash(&mut h);
    test_usize.hash(&mut h);
    test_i8.hash(&mut h);
    test_i16.hash(&mut h);
    test_i32.hash(&mut h);
    test_i64.hash(&mut h);
    test_i128.hash(&mut h);
    test_isize.hash(&mut h);

    // This depends on the hashing algorithm. See note at top of file.
    let expected = (2736651863462566372, 8121090595289675650);

    assert_eq!(h.finalize(), expected);
}

#[test]
fn test_hash_usize() {
    // Test that usize specifically is handled consistently across platforms.
    let test_usize = 0xABCDEF01_usize;

    let mut h = StableHasher::new();
    test_usize.hash(&mut h);

    // This depends on the hashing algorithm. See note at top of file.
    let expected = (5798740672699530587, 11186240177685111648);

    assert_eq!(h.finalize(), expected);
}

#[test]
fn test_hash_isize() {
    // Test that isize specifically is handled consistently across platforms.
    let test_isize = -7_isize;

    let mut h = StableHasher::new();
    test_isize.hash(&mut h);

    // This depends on the hashing algorithm. See note at top of file.
    let expected = (14721296605626097289, 11385941877786388409);

    assert_eq!(h.finalize(), expected);
}
