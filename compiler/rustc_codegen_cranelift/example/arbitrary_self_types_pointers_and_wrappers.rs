// Adapted from rustc run-pass test suite

#![feature(no_core, arbitrary_self_types, box_syntax)]
#![feature(rustc_attrs)]

#![feature(start, lang_items)]
#![no_core]

extern crate mini_core;

use mini_core::*;

macro_rules! assert_eq {
    ($l:expr, $r: expr) => {
        if $l != $r {
            panic(stringify!($l != $r));
        }
    }
}

struct Ptr<T: ?Sized>(Box<T>);

impl<T: ?Sized> Deref for Ptr<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &*self.0
    }
}

impl<T: Unsize<U> + ?Sized, U: ?Sized> CoerceUnsized<Ptr<U>> for Ptr<T> {}
impl<T: Unsize<U> + ?Sized, U: ?Sized> DispatchFromDyn<Ptr<U>> for Ptr<T> {}

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
    // This method isn't object-safe yet. Unsized by-value `self` is object-safe (but not callable
    // without unsized_locals), but wrappers arond `Self` currently are not.
    // FIXME (mikeyhew) uncomment this when unsized rvalues object-safety is implemented
    // fn wrapper(self: Wrapper<Self>) -> i32;
    fn ptr_wrapper(self: Ptr<Wrapper<Self>>) -> i32;
    fn wrapper_ptr(self: Wrapper<Ptr<Self>>) -> i32;
    fn wrapper_ptr_wrapper(self: Wrapper<Ptr<Wrapper<Self>>>) -> i32;
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
}

#[start]
fn main(_: isize, _: *const *const u8) -> isize {
    let pw = Ptr(box Wrapper(5)) as Ptr<Wrapper<dyn Trait>>;
    assert_eq!(pw.ptr_wrapper(), 5);

    let wp = Wrapper(Ptr(box 6)) as Wrapper<Ptr<dyn Trait>>;
    assert_eq!(wp.wrapper_ptr(), 6);

    let wpw = Wrapper(Ptr(box Wrapper(7))) as Wrapper<Ptr<Wrapper<dyn Trait>>>;
    assert_eq!(wpw.wrapper_ptr_wrapper(), 7);

    0
}
