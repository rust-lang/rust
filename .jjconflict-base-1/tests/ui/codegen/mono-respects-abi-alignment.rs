//! Test that monomorphization correctly distinguishes types with different ABI alignment.
//!
//! On x86_64-linux-gnu and similar platforms, structs get 8-byte "preferred"
//! alignment, but their "ABI" alignment (what actually matters for data layout)
//! is the largest alignment of any field. If monomorphization incorrectly uses
//! "preferred" alignment instead of "ABI" alignment, it might unify types `A`
//! and `B` even though `S<A>` and `S<B>` have field `t` at different offsets,
//! leading to incorrect method dispatch for `unwrap()`.

//@ run-pass

#[derive(Copy, Clone)]
struct S<T> {
    #[allow(dead_code)]
    i: u8,
    t: T,
}

impl<T> S<T> {
    fn unwrap(self) -> T {
        self.t
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
struct A((u32, u32)); // Different ABI alignment than B

#[derive(Copy, Clone, PartialEq, Debug)]
struct B(u64); // Different ABI alignment than A

pub fn main() {
    static CA: S<A> = S { i: 0, t: A((13, 104)) };
    static CB: S<B> = S { i: 0, t: B(31337) };

    assert_eq!(CA.unwrap(), A((13, 104)));
    assert_eq!(CB.unwrap(), B(31337));
}
