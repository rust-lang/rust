// Regression test for issue #76740.
//@ run-pass
//@ compile-flags: -Zmir-opt-level=4

#[derive(Copy, Clone)]
pub struct V([usize; 4]);

impl V {
    fn new() -> Self {
        V([0; 4])
    }

    #[inline(never)]
    fn check(mut self) {
        assert_eq!(self.0[0], 0);
        self.0[0] = 1;
    }
}

fn main() {
    let v = V::new();
    let mut i = 0;
    while i != 10 {
        // Copy propagation incorrectly assumed that Operand::Move does not
        // mutate the local, and used the same v for each V::check call,
        // rather than a copy.
        v.check();
        i += 1;
    }
}
