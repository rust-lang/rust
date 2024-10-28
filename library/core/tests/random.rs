use core::random::chacha::*;
use core::random::{DeterministicRandomSource, RandomSource};

// Test the quarter-round function.
#[test]
fn test_round() {
    let a = 0x11111111;
    let b = 0x01020304;
    let c = 0x9b8d6f43;
    let d = 0x01234567;
    let (a, b, c, d) = quarter_round(a, b, c, d);
    assert_eq!(a, 0xea2a92f4);
    assert_eq!(b, 0xcb1cf8ce);
    assert_eq!(c, 0x4581472e);
    assert_eq!(d, 0x5881c4bb);
}

// Test the block function.
// RFC 8439 only gives a test vector for 20 rounds, so we use that here.
#[test]
fn test_block() {
    let key = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
        0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
        0x1e, 0x1f,
    ];
    let counter_nonce = 0x00_00_00_00_4a_00_00_00_09_00_00_00_00_00_00_01;

    let block = block(&key, counter_nonce, 20);

    #[rustfmt::skip] // Preserve the formatting from the RFC.
    assert_eq!(block, [
        0x10, 0xf1, 0xe7, 0xe4, 0xd1, 0x3b, 0x59, 0x15, 0x50, 0x0f, 0xdd, 0x1f, 0xa3, 0x20, 0x71, 0xc4,
        0xc7, 0xd1, 0xf4, 0xc7, 0x33, 0xc0, 0x68, 0x03, 0x04, 0x22, 0xaa, 0x9a, 0xc3, 0xd4, 0x6c, 0x4e,
        0xd2, 0x82, 0x64, 0x46, 0x07, 0x9f, 0xaa, 0x09, 0x14, 0xc2, 0xd7, 0x05, 0xd9, 0x8b, 0x02, 0xa2,
        0xb5, 0x12, 0x9c, 0xd1, 0xde, 0x16, 0x4e, 0xb9, 0xcb, 0xd0, 0x83, 0xe8, 0xa2, 0x50, 0x3c, 0x4e,
    ]);
}

#[test]
fn test_source() {
    // Test that two sources produce the same output after a zero-sized request.
    let mut rng1 = DeterministicRandomSource::from_seed([42; 32]);
    let mut rng2 = DeterministicRandomSource::from_seed([42; 32]);
    rng1.fill_bytes(&mut []);
    let mut b1 = [0; 64];
    let mut b2 = [0; 64];
    rng1.fill_bytes(&mut b1);
    rng2.fill_bytes(&mut b2);
    assert_eq!(b1, b2);

    // Test that two sources with different seeds produce different data.
    let mut rng1 = DeterministicRandomSource::from_seed([b'A'; 32]);
    let mut rng2 = DeterministicRandomSource::from_seed([b'B'; 32]);
    let mut b1 = [0; 64]; // This size should be large enough to guarantee uniqueness.
    let mut b2 = [0; 64];
    rng1.fill_bytes(&mut b1);
    rng2.fill_bytes(&mut b2);
    assert_ne!(b1, b2);

    // Test that the source always generates the same bytes, irrespective of the
    // size of the individual requests.
    let mut rng1 = DeterministicRandomSource::from_seed([4; 32]);
    let mut rng2 = DeterministicRandomSource::from_seed([4; 32]);
    let mut b1 = [0; 128];
    let mut b2 = [0; 128];
    rng1.fill_bytes(&mut b1[0..63]);
    rng1.fill_bytes(&mut b1[63..128]);
    rng2.fill_bytes(&mut b2);
    assert_eq!(b1, b2);

    // Test the output of the RNG.
    let mut rng = DeterministicRandomSource::from_seed([42; 32]);
    let mut bytes = [0; 64];
    rng.fill_bytes(&mut bytes);
    assert_eq!(bytes, [
        182, 92, 255, 78, 146, 170, 167, 19, 201, 210, 224, 219, 84, 107, 104, 196, 224, 111, 107,
        198, 98, 121, 52, 115, 177, 219, 124, 69, 52, 111, 194, 63, 248, 202, 181, 174, 85, 116,
        46, 203, 37, 238, 86, 5, 123, 158, 53, 234, 229, 76, 64, 169, 181, 145, 115, 128, 64, 187,
        25, 75, 60, 217, 10, 169
    ]);
}
