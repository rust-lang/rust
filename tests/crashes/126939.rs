//@ known-bug: rust-lang/rust#126939

struct MySlice<T: Copy>(bool, T);
type MySliceBool = MySlice<[bool]>;

use std::mem;

struct P2<T> {
    a: T,
    b: MySliceBool,
}

macro_rules! check {
    ($t:ty, $align:expr) => ({
        assert_eq!(mem::align_of::<$t>(), $align);
    });
}

pub fn main() {
    check!(P2<u8>, 1);
}
