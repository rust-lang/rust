// Test that a method/associated non-method within an impl block of a concrete const type i.e. A<2>,
// is callable.
// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

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
