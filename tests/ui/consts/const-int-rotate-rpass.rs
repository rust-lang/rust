//@ run-pass

const LEFT: u32 = 0x10000b3u32.rotate_left(8);
const RIGHT: u32 = 0xb301u32.rotate_right(8);

// Rotating these should make no difference
//
// We test using 124 bits because to ensure that overlong bit shifts do
// not cause undefined behaviour. See #10183.
const LEFT_OVERFLOW: i16 = 0i16.rotate_left(124);
const RIGHT_OVERFLOW: i16 = 0i16.rotate_right(124);
const ONE_LEFT_OVERFLOW: u16 = 1u16.rotate_left(124);
const ONE_RIGHT_OVERFLOW: u16 = 1u16.rotate_right(124);

const NON_ZERO_LEFT_OVERFLOW: u16 = 0b10u16.rotate_left(124);
const NON_ZERO_RIGHT_OVERFLOW: u16 = 0b10u16.rotate_right(124);

// Rotating by 0 should have no effect
const ZERO_ROTATE_LEFT: i8 = 0b0010_0001i8.rotate_left(0);
const ZERO_ROTATE_RIGHT: i8 = 0b0111_1001i8.rotate_right(0);

// Rotating by a multiple of word size should also have no effect
const MULTIPLE_ROTATE_LEFT: i32 = 0b0010_0001i32.rotate_left(128);
const MULTIPLE_ROTATE_RIGHT: i32 = 0b0010_0001i32.rotate_right(128);

fn main() {
    assert_eq!(LEFT, 0xb301);
    assert_eq!(RIGHT, 0x0100_00b3);

    assert_eq!(LEFT_OVERFLOW, 0);
    assert_eq!(RIGHT_OVERFLOW, 0);
    assert_eq!(ONE_LEFT_OVERFLOW, 0b0001_0000_0000_0000);
    assert_eq!(ONE_RIGHT_OVERFLOW, 0b0001_0000);

    assert_eq!(NON_ZERO_LEFT_OVERFLOW, 0b0010_0000_0000_0000);
    assert_eq!(NON_ZERO_RIGHT_OVERFLOW, 0b0000_0000_0010_0000);

    assert_eq!(ZERO_ROTATE_LEFT, 0b0010_0001);
    assert_eq!(ZERO_ROTATE_RIGHT, 0b0111_1001);

    assert_eq!(MULTIPLE_ROTATE_LEFT, 0b0010_0001);
    assert_eq!(MULTIPLE_ROTATE_RIGHT, 0b0010_0001);
}
