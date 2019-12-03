// run-pass
// #![feature(const_int_pow)]
#![feature(wrapping_next_power_of_two)]

const IS_POWER_OF_TWO_A: bool = 0u32.is_power_of_two();
const IS_POWER_OF_TWO_B: bool = 32u32.is_power_of_two();
const IS_POWER_OF_TWO_C: bool = 33u32.is_power_of_two();
const IS_POWER_OF_TWO_D: bool = 3u8.is_power_of_two();

fn main() {
    assert!(!IS_POWER_OF_TWO_A);
    assert!(IS_POWER_OF_TWO_B);
    assert!(!IS_POWER_OF_TWO_C);
    assert!(!IS_POWER_OF_TWO_D);
}
