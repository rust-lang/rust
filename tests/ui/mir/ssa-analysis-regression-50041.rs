// build-pass
// compile-flags: -Z mir-opt-level=4

#![crate_type = "lib"]
#![feature(lang_items, rustc_attrs)]
#![no_std]

#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(1)]
#[rustc_nonnull_optimization_guaranteed]
struct NonNull<T: ?Sized>(*const T);

struct Unique<T: ?Sized>(NonNull<T>);

#[lang = "owned_box"]
pub struct Box<T: ?Sized>(Unique<T>);

impl<T: ?Sized> Drop for Box<T> {
    fn drop(&mut self) {}
}

#[lang = "box_free"]
#[inline(always)]
unsafe fn box_free<T: ?Sized>(ptr: Unique<T>) {
    dealloc(ptr.0.0 as _)
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
