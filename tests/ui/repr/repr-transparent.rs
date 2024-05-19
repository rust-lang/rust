// This file tests repr(transparent)-related errors reported during typeck. Other errors
// that are reported earlier and therefore preempt these are tested in:
// - repr-transparent-other-reprs.rs
// - repr-transparent-other-items.rs

#![feature(transparent_unions)]

use std::marker::PhantomData;

#[repr(transparent)]
struct NoFields;

#[repr(transparent)]
struct ContainsOnlyZst(());

#[repr(transparent)]
struct ContainsOnlyZstArray([bool; 0]);

#[repr(transparent)]
struct ContainsMultipleZst(PhantomData<*const i32>, NoFields);

#[repr(transparent)]
struct ContainsZstAndNonZst((), [i32; 2]);

#[repr(transparent)]
struct MultipleNonZst(u8, u8); //~ ERROR needs at most one field with non-trivial size or alignment

trait Mirror { type It: ?Sized; }
impl<T: ?Sized> Mirror for T { type It = Self; }

#[repr(transparent)]
pub struct StructWithProjection(f32, <f32 as Mirror>::It);
//~^ ERROR needs at most one field with non-trivial size or alignment

#[repr(transparent)]
struct NontrivialAlignZst(u32, [u16; 0]); //~ ERROR needs at most one field with non-trivial size or alignment

#[repr(align(32))]
struct ZstAlign32<T>(PhantomData<T>);

#[repr(transparent)]
struct GenericAlign<T>(ZstAlign32<T>, u32); //~ ERROR needs at most one field with non-trivial size or alignment

#[repr(transparent)]
struct WrapsZstWithAlignment([i32; 0]);

#[repr(transparent)] //~ ERROR unsupported representation for zero-variant enum
enum Void {} //~ ERROR transparent enum needs exactly one variant, but has 0

#[repr(transparent)]
enum FieldlessEnum {
    Foo,
}

#[repr(transparent)]
enum UnitFieldEnum {
    Foo(()),
}

#[repr(transparent)]
enum TooManyFieldsEnum {
    Foo(u32, String),
}
//~^^^ ERROR transparent enum needs at most one field with non-trivial size or alignment, but has 2

#[repr(transparent)]
enum MultipleVariants { //~ ERROR transparent enum needs exactly one variant, but has 2
    Foo(String),
    Bar,
}

#[repr(transparent)]
enum NontrivialAlignZstEnum { //~ ERROR needs at most one field with non-trivial size or alignment
    Foo(u32, [u16; 0]),
}

#[repr(transparent)]
enum GenericAlignEnum<T> { //~ ERROR needs at most one field with non-trivial size or alignment
    Foo { bar: ZstAlign32<T>, baz: u32 }
}

#[repr(transparent)]
union UnitUnion {
    u: (),
}

#[repr(transparent)]
union TooManyFields { //~ ERROR transparent union needs at most one field with non-trivial size or alignment, but has 2
    u: u32,
    s: i32
}

fn main() {}
