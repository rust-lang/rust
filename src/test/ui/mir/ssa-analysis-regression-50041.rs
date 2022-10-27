// build-pass
// compile-flags: -Z mir-opt-level=4

#![crate_type = "lib"]
#![feature(lang_items, ranged_int)]
#![no_std]

struct NonNull<T: ?Sized>(core::num::Ranged<*const T, { 1..=(usize::MAX as u128) }>);

struct Unique<T: ?Sized>(NonNull<T>);

#[lang = "owned_box"]
pub struct Box<T: ?Sized>(Unique<T>);

impl<T: ?Sized> Drop for Box<T> {
    fn drop(&mut self) {}
}

#[lang = "box_free"]
#[inline(always)]
unsafe fn box_free<T: ?Sized>(ptr: Unique<T>) {
    dealloc(ptr.0.0.get())
}

#[inline(never)]
fn dealloc<T: ?Sized>(_: *const T) {}

pub struct Foo<T>(T);

pub fn foo(a: Option<Box<Foo<usize>>>) -> usize {
    let f = match a {
        None => Foo(0),
        Some(vec) => *vec,
    };
    f.0
}
