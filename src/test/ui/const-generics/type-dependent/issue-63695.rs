// run-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

trait T {
    fn test<const A: i32>(&self) -> i32 { A }
}

struct S();

impl T for S {}

fn main() {
    let foo = S();
    assert_eq!(foo.test::<8i32>(), 8);
    assert_eq!(foo.test::<16i32>(), 16);
}
