#![allow(unreachable_code)]

use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;

pub struct Guard<T> {
    _phantom: PhantomData<T>,
}
impl<T> Deref for Guard<T> {
    type Target = T;
    fn deref(&self) -> &T {
        unimplemented!()
    }
}

pub struct DirectDeref<T>(T);
impl<T> Deref for DirectDeref<Arc<T>> {
    type Target = T;
    fn deref(&self) -> &T {
        unimplemented!()
    }
}

pub trait Access<T> {
    type Guard: Deref<Target = T>;
    fn load(&self) -> Self::Guard {
        unimplemented!()
    }
}
impl<T, A: Access<T>, P: Deref<Target = A>> Access<T> for P {
    //~^ NOTE: required for `Arc<ArcSwapAny<Arc<usize>>>` to implement `Access<_>`
    //~| NOTE unsatisfied trait bound introduced here
    type Guard = A::Guard;
}
impl<T> Access<T> for ArcSwapAny<T> {
    //~^ NOTE: multiple `impl`s satisfying `ArcSwapAny<Arc<usize>>: Access<_>` found
    type Guard = Guard<T>;
}
impl<T> Access<T> for ArcSwapAny<Arc<T>> {
    type Guard = DirectDeref<Arc<T>>;
}

pub struct ArcSwapAny<T> {
    _phantom_arc: PhantomData<T>,
}

pub fn foo() {
    let s: Arc<ArcSwapAny<Arc<usize>>> = unimplemented!();
    let guard: Guard<Arc<usize>> = s.load();
    //~^ ERROR: type annotations needed
    //~| HELP: try using a fully qualified path to specify the expected types
}

fn main() {}
