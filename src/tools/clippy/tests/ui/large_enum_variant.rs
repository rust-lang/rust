//@revisions: 32bit 64bit
//@aux-build:proc_macros.rs
//@no-rustfix
//@[32bit]ignore-bitwidth: 64
//@[64bit]ignore-bitwidth: 32
#![allow(dead_code)]
#![allow(unused_variables)]
#![warn(clippy::large_enum_variant)]

extern crate proc_macros;
use proc_macros::external;

enum LargeEnum {
    //~^ large_enum_variant
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
    //~^ large_enum_variant
    VariantOk(i32, u32),
    ContainingLargeEnum(LargeEnum),
}

enum LargeEnum3 {
    //~^ large_enum_variant
    ContainingMoreThanOneField(i32, [i32; 8000], [i32; 9500]),
    VoidVariant,
    StructLikeLittle { x: i32, y: i32 },
}

enum LargeEnum4 {
    //~^ large_enum_variant
    VariantOk(i32, u32),
    StructLikeLarge { x: [i32; 8000], y: i32 },
}

enum LargeEnum5 {
    //~^ large_enum_variant
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
    //~^ large_enum_variant
    A,
    B([u8; 1255]),
    C([u8; 200]),
}

enum LargeEnum8 {
    //~^ large_enum_variant
    VariantOk(i32, u32),
    ContainingMoreThanOneField([i32; 8000], [i32; 2], [i32; 9500], [i32; 30]),
}

enum LargeEnum9 {
    //~^ large_enum_variant
    A(Struct<()>),
    B(Struct2),
}

enum LargeEnumOk2<T> {
    //~^ large_enum_variant
    A(T),
    B(Struct2),
}

enum LargeEnumOk3<T> {
    //~^ large_enum_variant
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
    //~^ large_enum_variant
    A(bool),
    B([u64; 8000]),
}

enum ManuallyCopyLargeEnum {
    //~^ large_enum_variant
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
    //~^ large_enum_variant
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
    //~^ large_enum_variant
    Small,
    Large((T, [u8; 512])),
}

struct Foo<T> {
    foo: T,
}

enum WithGenerics {
    //~^ large_enum_variant
    Large([Foo<u64>; 64]),
    Small(u8),
}

enum PossiblyLargeEnumWithConst<const U: usize> {
    SmallBuffer([u8; 4]),
    MightyBuffer([u16; U]),
}

enum LargeEnumOfConst {
    //~^ large_enum_variant
    Ok,
    Error(PossiblyLargeEnumWithConst<256>),
}

enum WithRecursion {
    //~^ large_enum_variant
    Large([u64; 64]),
    Recursive(Box<WithRecursion>),
}

enum WithRecursionAndGenerics<T> {
    Large([T; 64]),
    Recursive(Box<WithRecursionAndGenerics<T>>),
}

enum LargeEnumWithGenericsAndRecursive {
    //~^ large_enum_variant
    Ok(),
    Error(WithRecursionAndGenerics<u64>),
}

fn main() {
    external!(
        enum LargeEnumInMacro {
            A(i32),
            B([i32; 8000]),
        }
    );
}

mod issue11915 {
    use std::sync::atomic::AtomicPtr;

    pub struct Bytes {
        ptr: *const u8,
        len: usize,
        // inlined "trait object"
        data: AtomicPtr<()>,
        vtable: &'static Vtable,
    }
    pub(crate) struct Vtable {
        /// fn(data, ptr, len)
        pub clone: unsafe fn(&AtomicPtr<()>, *const u8, usize) -> Bytes,
        /// fn(data, ptr, len)
        ///
        /// takes `Bytes` to value
        pub to_vec: unsafe fn(&AtomicPtr<()>, *const u8, usize) -> Vec<u8>,
        /// fn(data, ptr, len)
        pub drop: unsafe fn(&mut AtomicPtr<()>, *const u8, usize),
    }

    enum NoWarnings {
        //~[64bit]^ large_enum_variant
        BigBoi(PublishWithBytes),
        _SmallBoi(u8),
    }

    enum MakesClippyAngry {
        //~[64bit]^ large_enum_variant
        BigBoi(PublishWithVec),
        _SmallBoi(u8),
    }

    struct PublishWithBytes {
        _dup: bool,
        _retain: bool,
        _topic: Bytes,
        __topic: Bytes,
        ___topic: Bytes,
        ____topic: Bytes,
        _pkid: u16,
        _payload: Bytes,
        __payload: Bytes,
        ___payload: Bytes,
        ____payload: Bytes,
        _____payload: Bytes,
    }

    struct PublishWithVec {
        _dup: bool,
        _retain: bool,
        _topic: Vec<u8>,
        __topic: Vec<u8>,
        ___topic: Vec<u8>,
        ____topic: Vec<u8>,
        _pkid: u16,
        _payload: Vec<u8>,
        __payload: Vec<u8>,
        ___payload: Vec<u8>,
        ____payload: Vec<u8>,
        _____payload: Vec<u8>,
    }
}
