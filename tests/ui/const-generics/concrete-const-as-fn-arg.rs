// Test that a concrete const type i.e. A<2>, can be used as an argument type in a function
//@ run-pass

struct A<const N: usize>; // ok

fn with_concrete_const_arg(_: A<2>) -> u32 { 17 }

fn main() {
    let val: A<2> = A;
    assert_eq!(with_concrete_const_arg(val), 17);
}
