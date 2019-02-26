const NEGATIVE_A: bool = (-10i32).is_negative();
const NEGATIVE_B: bool = 10i32.is_negative();
const POSITIVE_A: bool= (-10i32).is_positive();
const POSITIVE_B: bool= 10i32.is_positive();

fn main() {
    assert!(NEGATIVE_A);
    assert!(!NEGATIVE_B);
    assert!(!POSITIVE_A);
    assert!(POSITIVE_B);
}
