// run-pass
// check-run-results

pub fn f() -> f64 {
    std::hint::black_box(-1.0) % std::hint::black_box(-1.0)
}

pub fn g() -> f64 {
    -1.0 % -1.0
}

pub fn main() {
    assert_eq!(-1, g().signum() as i32);
    assert_eq!((-0.0_f64).to_bits(), f().to_bits());
    assert_eq!(f().signum(), g().signum());
}
