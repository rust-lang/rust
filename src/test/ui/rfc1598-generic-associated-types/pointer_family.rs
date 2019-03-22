#![feature(generic_associated_types)]
//~^ WARNING the feature `generic_associated_types` is incomplete

// FIXME(#44265): "type argument not allowed" errors will be addressed in a follow-up PR.

use std::rc::Rc;
use std::sync::Arc;
use std::ops::Deref;

trait PointerFamily {
    type Pointer<T>: Deref<Target = T>;
    fn new<T>(value: T) -> Self::Pointer<T>;
    //~^ ERROR type arguments are not allowed for this type [E0109]
}

struct ArcFamily;

impl PointerFamily for ArcFamily {
    type Pointer<T> = Arc<T>;
    fn new<T>(value: T) -> Self::Pointer<T> {
    //~^ ERROR type arguments are not allowed for this type [E0109]
        Arc::new(value)
    }
}

struct RcFamily;

impl PointerFamily for RcFamily {
    type Pointer<T> = Rc<T>;
    fn new<T>(value: T) -> Self::Pointer<T> {
    //~^ ERROR type arguments are not allowed for this type [E0109]
        Rc::new(value)
    }
}

struct Foo<P: PointerFamily> {
    bar: P::Pointer<String>,
    //~^ ERROR type arguments are not allowed for this type [E0109]
}

fn main() {}
