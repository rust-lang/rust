// run-pass
//
// see issue #70529
#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

struct A<const N: usize> {
    arr: [u8; N],
}

impl<const N: usize> A<N> {
    fn new() -> Self {
        A {
            arr: [0; N],
        }
    }

    fn value(&self) -> usize {
        N
    }
}

fn main() {
    let a = A::new();
    let [_, _] = a.arr;
    assert_eq!(a.value(), 2);
}
