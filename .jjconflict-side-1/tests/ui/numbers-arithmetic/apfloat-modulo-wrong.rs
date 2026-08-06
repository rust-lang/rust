//@ run-pass
//@ check-run-results
// regression test for issue #109567

fn f() -> f64 {
    std::hint::black_box(-1.0) % std::hint::black_box(-1.0)
}

const G: f64 = -1.0 % -1.0;

pub fn main() {
    assert_eq!(-1, G.signum() as i32);
    assert_eq!((-0.0_f64).to_bits(), G.to_bits());
    assert_eq!(f().signum(), G.signum());
}
