//@ build-pass
//@ compile-flags: -Z mir-opt-level=4

#![crate_type = "lib"]
#![feature(lang_items)]
#![no_std]

struct NonNull<T: ?Sized>(*const T);

struct Unique<T: ?Sized>(NonNull<T>);

#[lang = "owned_box"]
pub struct Box<T: ?Sized>(Unique<T>);

impl<T: ?Sized> Drop for Box<T> {
    #[inline(always)]
    fn drop(&mut self) {
        dealloc(self.0.0.0)
    }
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
