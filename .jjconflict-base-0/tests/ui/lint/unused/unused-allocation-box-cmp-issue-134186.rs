//@ check-pass
// Regression test for <https://github.com/rust-lang/rust/issues/134186>.
// Comparing two `Box` values with `==` autorefs each `Box::new(...)` to `&Box<T>`, so the
// allocation is necessary and `unused_allocation` must not fire

#![deny(unused_allocation)]

pub fn main() {
    let a = Box::new(42);

    // `PartialEq for Box<T>` takes the operands by reference, so these allocations are used.
    let _ = a == Box::new(99);
    let _ = Box::new(99) == a;
    let _ = Box::new(1) == Box::new(2);
}
