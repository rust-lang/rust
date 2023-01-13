// run-pass

#![allow(non_upper_case_globals)]
#![allow(dead_code)]
/*!
 * On x86_64-linux-gnu and possibly other platforms, structs get 8-byte "preferred" alignment,
 * but their "ABI" alignment (i.e., what actually matters for data layout) is the largest alignment
 * of any field. (Also, `u64` has 8-byte ABI alignment; this is not always true).
 *
 * On such platforms, if monomorphize uses the "preferred" alignment, then it will unify
 * `A` and `B`, even though `S<A>` and `S<B>` have the field `t` at different offsets,
 * and apply the wrong instance of the method `unwrap`.
 */

#[derive(Copy, Clone)]
struct S<T> { i:u8, t:T }

impl<T> S<T> {
    fn unwrap(self) -> T {
        self.t
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
struct A((u32, u32));

#[derive(Copy, Clone, PartialEq, Debug)]
struct B(u64);

pub fn main() {
    static Ca: S<A> = S { i: 0, t: A((13, 104)) };
    static Cb: S<B> = S { i: 0, t: B(31337) };
    assert_eq!(Ca.unwrap(), A((13, 104)));
    assert_eq!(Cb.unwrap(), B(31337));
}
