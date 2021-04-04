// run-pass
//
// see issue #70529
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

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
