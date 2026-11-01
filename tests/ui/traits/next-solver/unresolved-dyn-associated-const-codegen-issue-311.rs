//@ compile-flags: -Znext-solver
//@ build-fail
//@ forbid-output: internal compiler error

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

fn main() {
    println!("{}", Of::<Object>::SIZE);
}

//@ normalize-stderr: "\n\n$" -> "\n"
