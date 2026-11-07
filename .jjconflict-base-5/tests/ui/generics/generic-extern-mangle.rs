//@ run-pass
use std::ops::Add;

extern "C" fn foo<T: Add>(a: T, b: T) -> T::Output { a + b }

fn main() {
    assert_eq!(100u8, foo(0u8, 100u8));
    assert_eq!(100u16, foo(0u16, 100u16));
}
