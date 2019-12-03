// run-pass
// #![feature(const_int_pow)]
#![feature(wrapping_next_power_of_two)]

const IS_POWER_OF_TWO_A: bool = 0u32.is_power_of_two();
const IS_POWER_OF_TWO_B: bool = 32u32.is_power_of_two();
const IS_POWER_OF_TWO_C: bool = 33u32.is_power_of_two();

const NEXT_POWER_OF_TWO_A: u32 = 0u32.next_power_of_two();
const NEXT_POWER_OF_TWO_B: u8 = 2u8.next_power_of_two();
const NEXT_POWER_OF_TWO_C: u8 = 3u8.next_power_of_two();
const NEXT_POWER_OF_TWO_D: u8 = 127u8.next_power_of_two();

const CHECKED_NEXT_POWER_OF_TWO_A: Option<u32> = 0u32.checked_next_power_of_two();
const CHECKED_NEXT_POWER_OF_TWO_B: Option<u8> = 2u8.checked_next_power_of_two();
const CHECKED_NEXT_POWER_OF_TWO_C: Option<u8> = 3u8.checked_next_power_of_two();
const CHECKED_NEXT_POWER_OF_TWO_D: Option<u8> = 127u8.checked_next_power_of_two();
const CHECKED_NEXT_POWER_OF_TWO_E: Option<u8> = 129u8.checked_next_power_of_two();

const WRAPPING_NEXT_POWER_OF_TWO_A: u32 = 0u32.wrapping_next_power_of_two();
const WRAPPING_NEXT_POWER_OF_TWO_B: u8 = 2u8.wrapping_next_power_of_two();
const WRAPPING_NEXT_POWER_OF_TWO_C: u8 = 3u8.wrapping_next_power_of_two();
const WRAPPING_NEXT_POWER_OF_TWO_D: u8 = 127u8.wrapping_next_power_of_two();
const WRAPPING_NEXT_POWER_OF_TWO_E: u8 = u8::max_value().wrapping_next_power_of_two();

fn main() {
    assert!(!IS_POWER_OF_TWO_A);
    assert!(IS_POWER_OF_TWO_B);
    assert!(!IS_POWER_OF_TWO_C);

    assert_eq!(NEXT_POWER_OF_TWO_A, 2);
    assert_eq!(NEXT_POWER_OF_TWO_B, 2);
    assert_eq!(NEXT_POWER_OF_TWO_C, 4);
    assert_eq!(NEXT_POWER_OF_TWO_D, 128);

    assert_eq!(CHECKED_NEXT_POWER_OF_TWO_A, Some(2));
    assert_eq!(CHECKED_NEXT_POWER_OF_TWO_B, Some(2));
    assert_eq!(CHECKED_NEXT_POWER_OF_TWO_C, Some(4));
    assert_eq!(CHECKED_NEXT_POWER_OF_TWO_D, Some(128));
    assert_eq!(CHECKED_NEXT_POWER_OF_TWO_E, None);

    assert_eq!(WRAPPING_NEXT_POWER_OF_TWO_A, 2);
    assert_eq!(WRAPPING_NEXT_POWER_OF_TWO_B, 2);
    assert_eq!(WRAPPING_NEXT_POWER_OF_TWO_C, 4);
    assert_eq!(WRAPPING_NEXT_POWER_OF_TWO_D, 128);
    assert_eq!(WRAPPING_NEXT_POWER_OF_TWO_E, 0);
}
