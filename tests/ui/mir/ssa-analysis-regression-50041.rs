//@ build-pass
//@ compile-flags: -Z mir-opt-level=4

#![crate_type = "lib"]
#![feature(lang_items)]
#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]
#![no_std]

struct NonNull<T>(*const T);

struct Unique<T>(NonNull<T>);

#[lang = "owned_box"]
pub struct Box<T>(Unique<T>);

impl<T> Drop for Box<T> {
    #[inline(always)]
    fn drop(&mut self) {
        dealloc(self.0.0.0)
    }
}

#[inline(never)]
fn dealloc<T>(_: *const T) {}

pub struct Foo<T>(T);

pub fn foo(a: Option<Box<Foo<usize>>>) -> usize {
    let f = match a {
        None => Foo(0),
        Some(vec) => *vec,
    };
    f.0
}
