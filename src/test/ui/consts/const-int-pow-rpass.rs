// run-pass

const IS_POWER_OF_TWO_A: bool = 0u32.is_power_of_two();
const IS_POWER_OF_TWO_B: bool = 32u32.is_power_of_two();
const IS_POWER_OF_TWO_C: bool = 33u32.is_power_of_two();

fn main() {
    assert!(!IS_POWER_OF_TWO_A);
    assert!(IS_POWER_OF_TWO_B);
    assert!(!IS_POWER_OF_TWO_C);
}
