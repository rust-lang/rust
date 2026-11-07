//@ run-pass
#![feature(arbitrary_self_types, unsize, coerce_unsized, dispatch_from_dyn)]
#![feature(rustc_attrs)]

use std::{
    cell::Cell,
    ops::{Deref, CoerceUnsized, DispatchFromDyn},
    marker::Unsize,
};

struct Ptr<T: ?Sized>(Box<T>);

impl<T: ?Sized> Deref for Ptr<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &*self.0
    }
}

impl<T: Unsize<U> + ?Sized, U: ?Sized> CoerceUnsized<Ptr<U>> for Ptr<T> {}
impl<T: Unsize<U> + ?Sized, U: ?Sized> DispatchFromDyn<Ptr<U>> for Ptr<T> {}


struct CellPtr<'a, T: ?Sized>(Cell<&'a T>);

impl<'a, T: ?Sized> Deref for CellPtr<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.0.get()
    }
}

impl<'a, T: Unsize<U> + ?Sized, U: ?Sized> CoerceUnsized<CellPtr<'a, U>> for CellPtr<'a, T> {}
impl<'a, T: Unsize<U> + ?Sized, U: ?Sized> DispatchFromDyn<CellPtr<'a, U>> for CellPtr<'a, T> {}

struct Wrapper<T: ?Sized>(T);

impl<T: ?Sized> Deref for Wrapper<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: CoerceUnsized<U>, U> CoerceUnsized<Wrapper<U>> for Wrapper<T> {}
impl<T: DispatchFromDyn<U>, U> DispatchFromDyn<Wrapper<U>> for Wrapper<T> {}


trait Trait {
    fn ptr_wrapper(self: Ptr<Wrapper<Self>>) -> i32;
    fn wrapper_ptr(self: Wrapper<Ptr<Self>>) -> i32;
    fn wrapper_ptr_wrapper(self: Wrapper<Ptr<Wrapper<Self>>>) -> i32;
    fn cell(self: CellPtr<Self>) -> i32;
}

impl Trait for i32 {
    fn ptr_wrapper(self: Ptr<Wrapper<Self>>) -> i32 {
        **self
    }
    fn wrapper_ptr(self: Wrapper<Ptr<Self>>) -> i32 {
        **self
    }
    fn wrapper_ptr_wrapper(self: Wrapper<Ptr<Wrapper<Self>>>) -> i32 {
        ***self
    }
    fn cell(self: CellPtr<Self>) -> i32 {
        *self
    }
}

fn main() {
    let pw = Ptr(Box::new(Wrapper(5))) as Ptr<Wrapper<dyn Trait>>;
    assert_eq!(pw.ptr_wrapper(), 5);

    let wp = Wrapper(Ptr(Box::new(6))) as Wrapper<Ptr<dyn Trait>>;
    assert_eq!(wp.wrapper_ptr(), 6);

    let wpw = Wrapper(Ptr(Box::new(Wrapper(7)))) as Wrapper<Ptr<Wrapper<dyn Trait>>>;
    assert_eq!(wpw.wrapper_ptr_wrapper(), 7);

    let c = CellPtr(Cell::new(&8)) as CellPtr<dyn Trait>;
    assert_eq!(c.cell(), 8);
}
