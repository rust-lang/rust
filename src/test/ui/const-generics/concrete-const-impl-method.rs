// Test that a method/associated non-method within an impl block of a concrete const type i.e. A<2>,
// is callable.
// run-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

pub struct A<const N: u32>;

impl A<2> {
    fn impl_method(&self) -> u32 {
        17
    }

    fn associated_non_method() -> u32 {
        17
    }
}

fn main() {
    let val: A<2> = A;
    assert_eq!(val.impl_method(), 17);
    assert_eq!(A::<2>::associated_non_method(), 17);
}
