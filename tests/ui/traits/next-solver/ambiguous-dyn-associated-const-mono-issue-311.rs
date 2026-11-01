//@ compile-flags: -Znext-solver
//@ check-fail

// A free const forces this evaluation before codegen.
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

const S: usize = Of::<Object>::SIZE;
fn main() {
    println!("{S}");
}

//@ normalize-stderr: "\n\n$" -> "\n"
