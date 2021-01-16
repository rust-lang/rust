// build-pass
// compile-flags: -Z mir-opt-level=3

#![crate_type="lib"]
#![feature(lang_items)]
#![no_std]

#[lang = "owned_box"]
pub struct Box<T: ?Sized>(*mut T);

impl<T: ?Sized> Drop for Box<T> {
    fn drop(&mut self) {
    }
}

#[lang = "box_free"]
#[inline(always)]
unsafe fn box_free<T: ?Sized>(ptr: *mut T) {
    dealloc(ptr)
}

#[inline(never)]
fn dealloc<T: ?Sized>(_: *mut T) {
}

pub struct Foo<T>(T);

pub fn foo(a: Option<Box<Foo<usize>>>) -> usize {
    let f = match a {
        None => Foo(0),
        Some(vec) => *vec,
    };
    f.0
}
