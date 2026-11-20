// Test codegen for unsize coercion when the source pointer is placed at memory.
//
// Previously, this will trigger ICE because cases where source pointer is placed on a memory
// location is not handled in `codegen_rvalue_operand()`. Such case can happen because C ABI might
// not treat a single field struct equivalently as the field type itself.

//@ build-pass

#![feature(unsize, coerce_unsized)]

use std::marker::Unsize;
use std::ops::CoerceUnsized;

#[repr(C)]
struct Ptr<T: ?Sized>(Box<T>);

impl<T: ?Sized, U: ?Sized> CoerceUnsized<Ptr<U>> for Ptr<T> where T: Unsize<U> {}

fn main() {
    let foo = Ptr(Box::new(5)) as Ptr<dyn std::any::Any>;
}
