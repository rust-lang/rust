// Test codegen for unsize coercion when the source pointer is placed at memory.

//@ build-pass

#![feature(coerce_unsized)]
#![feature(unsize)]

use std::marker::Unsize;
use std::ops::CoerceUnsized;

#[repr(C)]
struct Ptr<T: ?Sized>(Box<T>);
impl<T, U: ?Sized> CoerceUnsized<Ptr<U>> for Ptr<T> where T: Unsize<U> {}

fn main() {
    Ptr(Box::new(5)) as Ptr<dyn std::any::Any>;
}
