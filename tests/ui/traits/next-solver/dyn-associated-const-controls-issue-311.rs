//@ revisions: next old
//@[next] compile-flags: -Znext-solver
//@[old] compile-flags: -Znext-solver=coherence
//@ run-pass

// Matching object projections, builtin-only candidates, and generic
// associated constants must still work with either solver.
use std::marker::PhantomData;
use std::mem::size_of;

trait Blanket {
    type A;
}
impl<T: ?Sized> Blanket for T {
    type A = u8;
}
type Matching = dyn Blanket<A = u8>;

trait BuiltinOnly {
    type A;
}
type BuiltinObject = dyn BuiltinOnly<A = u16>;

struct Of<T: ?Sized>(PhantomData<T>);
impl<T: ?Sized + Blanket> Of<T> {
    const SIZE: usize = size_of::<T::A>();
}

const _: () = assert!(Of::<Matching>::SIZE == 1);
const _: () = assert!(size_of::<<Matching as Blanket>::A>() == 1);
const _: () = assert!(size_of::<<BuiltinObject as BuiltinOnly>::A>() == 2);

fn generic<T: Blanket + ?Sized>() -> usize {
    Of::<T>::SIZE
}
fn main() {
    // Generic associated constants must still evaluate after monomorphization.
    assert_eq!(generic::<u8>(), 1);
    assert_eq!(generic::<u16>(), 1);
}
