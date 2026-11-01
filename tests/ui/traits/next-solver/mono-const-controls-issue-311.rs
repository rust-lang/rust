//@ revisions: next old
//@[next] compile-flags: -Znext-solver
//@[old] compile-flags: -Znext-solver=coherence
//@ run-pass

// Generic constants may remain unevaluated until their arguments are known.
// The new error path must not reject these cases at monomorphization.
use std::mem::size_of;

trait Project {
    type Item;
}
impl<T: ?Sized> Project for T {
    type Item = u8;
}

fn size_of_projected<T: ?Sized + Project>() -> usize {
    size_of::<T::Item>()
}

fn repeat<const N: usize>() -> usize {
    [0u8; N].len()
}

const ONE: usize = size_of::<u8>();

fn main() {
    assert_eq!(size_of_projected::<u32>(), 1);
    assert_eq!(size_of_projected::<u64>(), 1);
    assert_eq!(repeat::<3>(), 3);
    assert_eq!(repeat::<16>(), 16);
    assert_eq!(ONE, 1);
}
