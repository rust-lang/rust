// Test that a concrete const type i.e. A<2>, can be used as an argument type in a function
// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

struct A<const N: usize>; // ok

fn with_concrete_const_arg(_: A<2>) -> u32 { 17 }

fn main() {
    let val: A<2> = A;
    assert_eq!(with_concrete_const_arg(val), 17);
}
