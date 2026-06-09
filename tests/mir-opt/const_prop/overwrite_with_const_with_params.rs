//@ test-mir-pass: GVN
//@ compile-flags: -O

// Regression test for https://github.com/rust-lang/rust/issues/118328

#![allow(unused_assignments)]

struct SizeOfConst<T>(std::marker::PhantomData<T>);
impl<T> SizeOfConst<T> {
    const SIZE: usize = std::mem::size_of::<T>();
}

// EMIT_MIR overwrite_with_const_with_params.size_of.GVN.diff
fn size_of<T>() -> usize {
    // CHECK-LABEL: fn size_of(
    // CHECK: _1 = const 0_usize;
    // CHECK-NEXT: _1 = const SizeOfConst::<T>::SIZE;
    // CHECK-NEXT: _0 = copy _1;
    let mut a = 0;
    a = SizeOfConst::<T>::SIZE;
    a
}

fn main() {
    assert_eq!(size_of::<u32>(), std::mem::size_of::<u32>());
}
