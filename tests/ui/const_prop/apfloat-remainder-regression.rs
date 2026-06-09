//@ run-pass
//@ compile-flags: -O -Zmir-opt-level=3 -Cno-prepopulate-passes

// Regression test for a broken MIR optimization (issue #102403).
pub fn f() -> f64 {
    std::hint::black_box(-1.0) % std::hint::black_box(-1.0)
}

pub fn g() -> f64 {
    -1.0 % -1.0
}

pub fn main() {
    assert_eq!(f().signum(), g().signum());
}
