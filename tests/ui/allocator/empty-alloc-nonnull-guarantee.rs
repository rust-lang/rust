//! Test that empty allocations produce non-null pointers
//!
//! This test ensures that Rust's allocator maintains the invariant that
//! Box<T> is always non-null, even for zero-sized types and empty allocations.
//! This is crucial for memory safety guarantees in Rust.

//@ run-pass

pub fn main() {
    assert!(Some(Box::new(())).is_some());

    let xs: Box<[()]> = Box::<[(); 0]>::new([]);
    assert!(Some(xs).is_some());

    struct Foo;
    assert!(Some(Box::new(Foo)).is_some());

    let ys: Box<[Foo]> = Box::<[Foo; 0]>::new([]);
    assert!(Some(ys).is_some());
}
