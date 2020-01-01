#![allow(incomplete_features)]
#![feature(generic_associated_types)]

// FIXME(#44265): allow type-generic associated types.

use std::rc::Rc;
use std::sync::Arc;
use std::ops::Deref;

trait PointerFamily {
    type Pointer<T>: Deref<Target = T>;
    //~^ ERROR type-generic associated types are not yet implemented
    fn new<T>(value: T) -> Self::Pointer<T>;
}

struct ArcFamily;

impl PointerFamily for ArcFamily {
    type Pointer<T> = Arc<T>;
    fn new<T>(value: T) -> Self::Pointer<T> {
        Arc::new(value)
    }
}

struct RcFamily;

impl PointerFamily for RcFamily {
    type Pointer<T> = Rc<T>;
    fn new<T>(value: T) -> Self::Pointer<T> {
        Rc::new(value)
    }
}

struct Foo<P: PointerFamily> {
    bar: P::Pointer<String>,
}

fn main() {}
