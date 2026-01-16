//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
//@ run-pass
#![expect(incomplete_features)]
#![feature(field_projections)]

use std::field::{Field, field_of};
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

fn main() {
    assert_eq!(<field_of!(Struct, a)>::OFFSET, 0);
    assert_eq!(<field_of!(Struct, b)>::OFFSET, 8);

    assert_eq!(<field_of!(Union, a)>::OFFSET, 0);
    assert_eq!(<field_of!(Union, b)>::OFFSET, 0);

    assert_eq!(<field_of!(Enum, A.a)>::OFFSET, 8);
    assert_eq!(<field_of!(Enum, A.b)>::OFFSET, 16);
    assert_eq!(<field_of!(Enum, B.x)>::OFFSET, 8);
    assert_eq!(<field_of!(Enum, B.y)>::OFFSET, 16);
    let s = Struct { a: 42, b: 24 };
    let r = &s;
    let a = project_ref::<Struct, field_of!(Struct, a)>(r);
    let b = project_ref::<Struct, field_of!(Struct, b)>(r);
    assert_eq!(*a, 42);
    assert_eq!(*b, 24);
}
