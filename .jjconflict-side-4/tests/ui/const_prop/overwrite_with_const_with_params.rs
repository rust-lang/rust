//@ compile-flags: -O
//@ run-pass

// Regression test for https://github.com/rust-lang/rust/issues/118328

#![allow(unused_assignments)]

struct SizeOfConst<T>(std::marker::PhantomData<T>);
impl<T> SizeOfConst<T> {
    const SIZE: usize = std::mem::size_of::<T>();
}

fn size_of<T>() -> usize {
    let mut a = 0;
    a = SizeOfConst::<T>::SIZE;
    a
}

fn main() {
    assert_eq!(size_of::<u32>(), std::mem::size_of::<u32>());
}
