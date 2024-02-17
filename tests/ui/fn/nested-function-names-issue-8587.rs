//@ run-pass
// Make sure nested functions are separate, even if they have
// equal name.
//
// Issue #8587


pub struct X;

impl X {
    fn f(&self) -> isize {
        #[inline(never)]
        fn inner() -> isize {
            0
        }
        inner()
    }

    fn g(&self) -> isize {
        #[inline(never)]
        fn inner_2() -> isize {
            1
        }
        inner_2()
    }

    fn h(&self) -> isize {
        #[inline(never)]
        fn inner() -> isize {
            2
        }
        inner()
    }
}

pub fn main() {
    let n = X;
    assert_eq!(n.f(), 0);
    assert_eq!(n.g(), 1);
    // This test `h` used to fail.
    assert_eq!(n.h(), 2);
}
