// revisions: full min

#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(full, feature(const_generics))]

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
//~^ ERROR cannot provide explicit generic arguments when `impl Trait` is used in argument position
}
