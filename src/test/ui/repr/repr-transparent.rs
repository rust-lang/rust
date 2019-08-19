// This file tests repr(transparent)-related errors reported during typeck. Other errors
// that are reported earlier and therefore preempt these are tested in:
// - repr-transparent-other-reprs.rs
// - repr-transparent-other-items.rs

#![feature(repr_align, transparent_enums, transparent_unions)]

use std::marker::PhantomData;

#[repr(transparent)]
struct NoFields; //~ ERROR needs exactly one non-zero-sized field

#[repr(transparent)]
struct ContainsOnlyZst(()); //~ ERROR needs exactly one non-zero-sized field

#[repr(transparent)]
struct ContainsOnlyZstArray([bool; 0]); //~ ERROR needs exactly one non-zero-sized field

#[repr(transparent)]
struct ContainsMultipleZst(PhantomData<*const i32>, NoFields);
//~^ ERROR needs exactly one non-zero-sized field

#[repr(transparent)]
struct MultipleNonZst(u8, u8); //~ ERROR needs exactly one non-zero-sized field

trait Mirror { type It: ?Sized; }
impl<T: ?Sized> Mirror for T { type It = Self; }

#[repr(transparent)]
pub struct StructWithProjection(f32, <f32 as Mirror>::It);
//~^ ERROR needs exactly one non-zero-sized field

#[repr(transparent)]
struct NontrivialAlignZst(u32, [u16; 0]); //~ ERROR alignment larger than 1

#[repr(align(32))]
struct ZstAlign32<T>(PhantomData<T>);

#[repr(transparent)]
struct GenericAlign<T>(ZstAlign32<T>, u32); //~ ERROR alignment larger than 1

#[repr(transparent)] //~ ERROR unsupported representation for zero-variant enum
enum Void {}
//~^ ERROR transparent enum needs exactly one variant, but has 0

#[repr(transparent)]
enum FieldlessEnum { //~ ERROR transparent enum needs exactly one non-zero-sized field, but has 0
    Foo,
}

#[repr(transparent)]
enum TooManyFieldsEnum {
    Foo(u32, String),
}
//~^^^ ERROR transparent enum needs exactly one non-zero-sized field, but has 2

#[repr(transparent)]
enum TooManyVariants { //~ ERROR transparent enum needs exactly one variant, but has 2
    Foo(String),
    Bar,
}

#[repr(transparent)]
union UnitUnion { //~ ERROR transparent union needs exactly one non-zero-sized field, but has 0
    u: (),
}

#[repr(transparent)]
union TooManyFields { //~ ERROR transparent union needs exactly one non-zero-sized field, but has 2
    u: u32,
    s: i32
}

fn main() {}
