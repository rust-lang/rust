// run-pass

#![feature(bindings_after_at)]
//~^ WARN the feature `bindings_after_at` is incomplete and may cause the compiler to crash

struct A { a: u8, b: u8 }

pub fn main() {
    match (A { a: 10, b: 20 }) {
        ref x @ A { ref a, b: 20 } => {
            assert_eq!(x.a, 10);
            assert_eq!(*a, 10);
        }
        A { b: ref _b, .. } => panic!(),
    }
}
