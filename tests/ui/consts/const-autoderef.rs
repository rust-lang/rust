//@ run-pass

const A: [u8; 1] = ['h' as u8];
const B: u8 = (&A)[0];
const C: &'static &'static &'static &'static [u8; 1] = & & & &A;
const D: u8 = (&C)[0];

pub fn main() {
    assert_eq!(B, A[0]);
    assert_eq!(D, A[0]);
}
