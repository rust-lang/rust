// run-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

struct A;
impl A {
    fn foo<const N: usize>() -> usize { N + 1 }
}

fn main() {
    assert_eq!(A::foo::<7>(), 8);
}
