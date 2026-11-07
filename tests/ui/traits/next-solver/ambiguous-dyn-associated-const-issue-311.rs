//@ compile-flags: -Znext-solver
//@ check-fail

// The builtin trait-object impl overlaps the blanket impl below.
// Neither assertion should be accepted when the projection is ambiguous.
use std::marker::PhantomData;
use std::mem::size_of;

trait Ob {
    type A;
}
impl<T: ?Sized> Ob for T {
    type A = u8;
}
struct Of<T: ?Sized>(PhantomData<T>);
impl<T: ?Sized + Ob> Of<T> {
    const SIZE: usize = size_of::<T::A>(); //~ ERROR cannot evaluate associated constant with unresolved type dependencies
}
type Object = dyn Ob<A = u16>;

const _: () = assert!(Of::<Object>::SIZE == 1);
const _: () = assert!(Of::<Object>::SIZE == 2);

fn main() {}

//@ normalize-stderr: "\n\n$" -> "\n"
