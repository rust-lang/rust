// issue: rust-lang/rust#106874
// ICE BoundUniversalRegionError

use std::marker::PhantomData;
use std::rc::Rc;

pub fn func<V, F: Fn(&mut V)>(f: F) -> A<impl X> {
    A(B(C::new(D::new(move |st| f(st)))))
    //~^ ERROR implementation of `FnOnce` is not general enough
    //~| ERROR implementation of `Fn` is not general enough
    //~| ERROR implementation of `FnOnce` is not general enough
    //~| ERROR implementation of `FnOnce` is not general enough
    //~| ERROR implementation of `Fn` is not general enough
    //~| ERROR implementation of `FnOnce` is not general enough
    //~| ERROR implementation of `Fn` is not general enough
    //~| ERROR implementation of `FnOnce` is not general enough
    //~| ERROR higher-ranked subtype error
    //~| ERROR higher-ranked subtype error
}

trait X {}
trait Y {
    type V;
}

struct A<T>(T);

struct B<T>(Rc<T>);
impl<T> X for B<T> {}

struct C<T: Y>(T::V);
impl<T: Y> C<T> {
    fn new(_: T) -> Rc<Self> {
        todo!()
    }
}
struct D<V, F>(F, PhantomData<fn(&mut V)>);

impl<V, F> D<V, F> {
    fn new(_: F) -> Self {
        todo!()
    }
}
impl<V, F: Fn(&mut V)> Y for D<V, F> {
    type V = V;
}

pub fn main() {}
