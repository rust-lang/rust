//@ revisions: old next generic_const_args
//@ [next] compile-flags: -Znext-solver
//@ [generic_const_args] compile-flags: -Znext-solver
//@ run-pass
#![expect(incomplete_features)]
#![feature(field_projections)]
#![cfg_attr(generic_const_args, feature(generic_const_args, min_generic_const_args))]

use std::field::{Field, field_of};
use std::mem::offset_of;
use std::ptr;

#[repr(C)]
pub struct Struct {
    a: i32,
    b: i64,
}

// FIXME(FRTs): need to mark these fields as used by the `field_of!` macro.
#[expect(dead_code)]
pub union Union {
    a: i32,
    b: i64,
}

#[repr(C, i8)]
pub enum Enum {
    A { a: i32, b: i64 },
    B { x: i64, y: i32 },
}

fn project_ref<'a, T, F: Field<Base = T>>(r: &'a T) -> &'a F::Type {
    unsafe { &*ptr::from_ref(r).byte_add(F::OFFSET).cast() }
}

#[repr(C)]
struct Generic<T> {
    a: u8,
    b: T,
}

fn generic_offset<T>() -> usize {
    <field_of!(Generic<T>, b)>::OFFSET
}

fn main() {
    assert_eq!(<field_of!(Struct, a)>::OFFSET, offset_of!(Struct, a));
    assert_eq!(<field_of!(Struct, b)>::OFFSET, offset_of!(Struct, b));
    assert_eq!(generic_offset::<u8>(), offset_of!(Generic<u8>, b));
    assert_eq!(generic_offset::<u64>(), offset_of!(Generic<u64>, b));

    let _: field_of!(Union, a);
    let _: field_of!(Union, b);

    let _: field_of!(Enum, A.a);
    let _: field_of!(Enum, A.b);
    let _: field_of!(Enum, B.x);
    let _: field_of!(Enum, B.y);

    let s = Struct { a: 42, b: 24 };
    let r = &s;
    let a = project_ref::<Struct, field_of!(Struct, a)>(r);
    let b = project_ref::<Struct, field_of!(Struct, b)>(r);
    assert_eq!(*a, 42);
    assert_eq!(*b, 24);
}
