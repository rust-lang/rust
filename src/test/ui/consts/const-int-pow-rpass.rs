// run-pass

#![feature(wrapping_next_power_of_two)]

const IS_POWER_OF_TWO_A: bool = 0u32.is_power_of_two();
const IS_POWER_OF_TWO_B: bool = 32u32.is_power_of_two();
const IS_POWER_OF_TWO_C: bool = 33u32.is_power_of_two();

const POW: u8 = 3u8.pow(5);

const CHECKED_POW_OK: Option<u8> = 3u8.checked_pow(5);
const CHECKED_POW_OVERFLOW: Option<u8> = 3u8.checked_pow(6);

const WRAPPING_POW: u8 = 3u8.wrapping_pow(6);
const OVERFLOWING_POW: (u8, bool) = 3u8.overflowing_pow(6);
const SATURATING_POW: u8 = 3u8.saturating_pow(6);

const NEXT_POWER_OF_TWO: u32 = 3u32.next_power_of_two();

const CHECKED_NEXT_POWER_OF_TWO_OK: Option<u32> = 3u32.checked_next_power_of_two();
const CHECKED_NEXT_POWER_OF_TWO_OVERFLOW: Option<u32> =
    u32::MAX.checked_next_power_of_two();

const WRAPPING_NEXT_POWER_OF_TWO: u32 =
    u32::MAX.wrapping_next_power_of_two();

fn main() {
    assert!(!IS_POWER_OF_TWO_A);
    assert!(IS_POWER_OF_TWO_B);
    assert!(!IS_POWER_OF_TWO_C);

    assert_eq!(POW, 243);

    assert_eq!(CHECKED_POW_OK, Some(243));
    assert_eq!(CHECKED_POW_OVERFLOW, None);

    assert_eq!(WRAPPING_POW, 217);
    assert_eq!(OVERFLOWING_POW, (217, true));
    assert_eq!(SATURATING_POW, u8::MAX);

    assert_eq!(NEXT_POWER_OF_TWO, 4);

    assert_eq!(CHECKED_NEXT_POWER_OF_TWO_OK, Some(4));
    assert_eq!(CHECKED_NEXT_POWER_OF_TWO_OVERFLOW, None);

    assert_eq!(WRAPPING_NEXT_POWER_OF_TWO, 0);
}
