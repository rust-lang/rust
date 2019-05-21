// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

struct S<const X: u32>;

impl<const X: u32> S<{X}> {
    fn x() -> u32 {
        X
    }
}

fn main() {
    assert_eq!(S::<19>::x(), 19);
}
