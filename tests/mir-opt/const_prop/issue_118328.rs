// unit-test: ConstProp
// compile-flags: -O
// skip-filecheck
#![allow(unused_assignments)]

struct SizeOfConst<T>(std::marker::PhantomData<T>);
impl<T> SizeOfConst<T> {
    const SIZE: usize = std::mem::size_of::<T>();
}

// EMIT_MIR issue_118328.size_of.ConstProp.diff
fn size_of<T>() -> usize {
    let mut a = 0;
    a = SizeOfConst::<T>::SIZE;
    a
}

fn main() {
    assert_eq!(size_of::<u32>(), std::mem::size_of::<u32>());
}
