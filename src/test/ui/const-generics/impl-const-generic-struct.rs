// run-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

struct S<const X: u32>;

impl<const X: u32> S<X> {
    fn x() -> u32 {
        X
    }
}

fn main() {
    assert_eq!(S::<19>::x(), 19);
}
