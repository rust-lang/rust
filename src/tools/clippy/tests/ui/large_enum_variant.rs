// aux-build:proc_macros.rs

#![allow(dead_code)]
#![allow(unused_variables)]
#![warn(clippy::large_enum_variant)]

extern crate proc_macros;
use proc_macros::external;

enum LargeEnum {
    A(i32),
    B([i32; 8000]),
}

enum GenericEnumOk<T> {
    A(i32),
    B([T; 8000]),
}

enum GenericEnum2<T> {
    A(i32),
    B([i32; 8000]),
    C(T, [i32; 8000]),
}

trait SomeTrait {
    type Item;
}

enum LargeEnumGeneric<A: SomeTrait> {
    Var(A::Item),
}

enum LargeEnum2 {
    VariantOk(i32, u32),
    ContainingLargeEnum(LargeEnum),
}

enum LargeEnum3 {
    ContainingMoreThanOneField(i32, [i32; 8000], [i32; 9500]),
    VoidVariant,
    StructLikeLittle { x: i32, y: i32 },
}

enum LargeEnum4 {
    VariantOk(i32, u32),
    StructLikeLarge { x: [i32; 8000], y: i32 },
}

enum LargeEnum5 {
    VariantOk(i32, u32),
    StructLikeLarge2 { x: [i32; 8000] },
}

enum LargeEnumOk {
    LargeA([i32; 8000]),
    LargeB([i32; 8001]),
}

enum LargeEnum6 {
    A,
    B([u8; 255]),
    C([u8; 200]),
}

enum LargeEnum7 {
    A,
    B([u8; 1255]),
    C([u8; 200]),
}

enum LargeEnum8 {
    VariantOk(i32, u32),
    ContainingMoreThanOneField([i32; 8000], [i32; 2], [i32; 9500], [i32; 30]),
}

enum LargeEnum9 {
    A(Struct<()>),
    B(Struct2),
}

enum LargeEnumOk2<T> {
    A(T),
    B(Struct2),
}

enum LargeEnumOk3<T> {
    A(Struct<T>),
    B(Struct2),
}

struct Struct<T> {
    a: i32,
    t: T,
}

struct Struct2 {
    a: [i32; 8000],
}

#[derive(Copy, Clone)]
enum CopyableLargeEnum {
    A(bool),
    B([u64; 8000]),
}

enum ManuallyCopyLargeEnum {
    A(bool),
    B([u64; 8000]),
}

impl Clone for ManuallyCopyLargeEnum {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for ManuallyCopyLargeEnum {}

enum SomeGenericPossiblyCopyEnum<T> {
    A(bool, std::marker::PhantomData<T>),
    B([u64; 4000]),
}

impl<T: Copy> Clone for SomeGenericPossiblyCopyEnum<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Copy> Copy for SomeGenericPossiblyCopyEnum<T> {}

enum LargeEnumWithGenerics<T> {
    Small,
    Large((T, [u8; 512])),
}

struct Foo<T> {
    foo: T,
}

enum WithGenerics {
    Large([Foo<u64>; 64]),
    Small(u8),
}

enum PossiblyLargeEnumWithConst<const U: usize> {
    SmallBuffer([u8; 4]),
    MightyBuffer([u16; U]),
}

enum LargeEnumOfConst {
    Ok,
    Error(PossiblyLargeEnumWithConst<256>),
}

fn main() {
    external!(
        enum LargeEnumInMacro {
            A(i32),
            B([i32; 8000]),
        }
    );
}
