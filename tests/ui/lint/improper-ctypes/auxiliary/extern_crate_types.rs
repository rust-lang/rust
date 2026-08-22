/// a bank of types (structs, function pointers) that are safe or unsafe for whatever reason,
/// with or without said unsafety being explicitely ignored

#[repr(C)]
pub struct SafeStruct (pub i32);

#[repr(C)]
pub struct UnsafeStruct (pub String);

#[repr(C)]
pub struct AllowedUnsafeStruct (pub String);

// refs are only unsafe if the value comes from the other side of the FFI boundary
// due to the non-null assumption
// (technically there are also assumptions about non-dandling, alignment, aliasing,
//  lifetimes, etc...)
#[repr(C)]
pub struct UnsafeFromForeignStruct<'a> (pub &'a u32);

#[repr(C)]
pub struct AllowedUnsafeFromForeignStruct<'a> (pub &'a u32);


pub type SafeFnPtr = extern "C" fn(i32)->i32;

pub type UnsafeFnPtr = extern "C" fn((i32,i32))->i32;

#[allow(improper_ctypes)]
pub type AllowedUnsafeFnPtr = extern "C" fn(&[i32])->i32;

pub type UnsafeRustCalleeFnPtr = extern "C" fn(i32)->&'static i32;

#[allow(improper_ctypes)]
pub type AllowedUnsafeRustCalleeFnPtr = extern "C" fn(i32)->&'static i32;

pub type UnsafeForeignCalleeFnPtr = extern "C" fn(&i32);

#[allow(improper_ctypes)]
pub type AllowedUnsafeForeignCalleeFnPtr = extern "C" fn(&i32);


// ////////////////////////////////////
/// types used in specific issue-based tests that need extern-crate types

#[repr(C)]
#[non_exhaustive]
pub struct NonExhaustiveStruct {
    pub field: u8
}

#[repr(C)]
#[non_exhaustive]
pub enum NonExhaustiveEnum {
    FirstVariant,
    SecondVariant(u8),
}

#[repr(C)]
#[non_exhaustive]
pub enum NonExhaustiveEnumWithBadVariant {
    Variant1,
    Variant2(()),
}

#[repr(C)]
pub enum NonExhaustiveEnumVariant {
    Variant1,
    #[non_exhaustive]
    Variant2((u32,)),
}

#[repr(C)]
#[non_exhaustive]
pub enum NonExhaustivePureEnum {
    Variant1,
    Variant2,
}

#[non_exhaustive]
pub enum NonExhaustiveOptionLike {
    NoneVariant,
    SomeVariant(::std::num::NonZeroUsize),
}

extern "C" {
    pub fn nonexhaustivestruct_create() -> *mut NonExhaustiveStruct;
    pub fn nonexhaustivestruct_destroy(s: *mut NonExhaustiveStruct);
    pub fn nonexhaustiveenum_create() -> *mut NonExhaustiveEnum;
    pub fn nonexhaustiveenum_destroy(s: *mut NonExhaustiveEnum);
    pub fn nonexhaustiveenumwbv_create() -> *mut NonExhaustiveEnumWithBadVariant;
    pub fn nonexhaustiveenumwbv_destroy(s: *mut NonExhaustiveEnumWithBadVariant);
    pub fn nonexhaustiveenumvariant_create() -> *mut NonExhaustiveEnumVariant;
    pub fn nonexhaustiveenumvariant_destroy(s: *mut NonExhaustiveEnumVariant);

    pub fn nonexhaustivestruct_onstack() -> NonExhaustiveStruct;
    pub fn nonexhaustivestruct_owned() -> NonExhaustiveStruct;
}
