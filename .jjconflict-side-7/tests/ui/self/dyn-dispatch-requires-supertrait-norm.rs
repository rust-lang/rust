//@ check-pass

#![feature(derive_coerce_pointee)]
#![feature(arbitrary_self_types)]

use std::ops::Deref;
use std::marker::CoercePointee;
use std::sync::Arc;

trait MyTrait<T> {}

#[derive(CoercePointee)]
#[repr(transparent)]
struct MyArc<T: ?Sized + MyTrait<u8>>(Arc<T>);

impl<T: ?Sized + MyTrait<u8>> Deref for MyArc<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

trait Mirror {
    type Assoc;
}
impl<T> Mirror for T {
    type Assoc = T;
}

// This is variant on "tests/ui/self/dyn-dispatch-requires-supertrait.rs" but with
// a supertrait that requires normalization to match the pred in the old solver.
trait MyOtherTrait: MyTrait<<u8 as Mirror>::Assoc> {
    fn foo(self: MyArc<Self>);
}

fn test(_: MyArc<dyn MyOtherTrait>) {}

fn main() {}
