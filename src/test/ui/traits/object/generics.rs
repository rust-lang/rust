// run-pass
// test for #8664

use std::marker;

pub trait Trait2<A> {
    fn doit(&self) -> A;
}

pub struct Impl<A1, A2, A3> {
    m1: marker::PhantomData<(A1,A2,A3)>,
    /*
     * With A2 we get the ICE:
     * task <unnamed> failed at 'index out of bounds: the len is 1 but the index is 1',
     * src/librustc/middle/subst.rs:58
     */
    t: Box<dyn Trait2<A2>+'static>
}

impl<A1, A2, A3> Impl<A1, A2, A3> {
    pub fn step(&self) {
        self.t.doit();
    }
}

// test for #8601

enum Type<T> { Constant(T) }

trait Trait<K,V> {
    fn method(&self, _: Type<(K,V)>) -> isize;
}

impl<V> Trait<u8,V> for () {
    fn method(&self, _x: Type<(u8,V)>) -> isize { 0 }
}

pub fn main() {
    let a = Box::new(()) as Box<dyn Trait<u8, u8>>;
    assert_eq!(a.method(Type::Constant((1, 2))), 0);
}
