//@ check-pass

trait Usizer {
    fn m(self) -> usize;
}

fn f<const N: usize>(u: impl Usizer) -> usize {
    N + u.m()
}

struct Usizable;

impl Usizer for Usizable {
    fn m(self) -> usize {
        16
    }
}

fn main() {
    assert_eq!(f::<4usize>(Usizable), 20usize);
}
