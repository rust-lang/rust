/// a bank of types (structs, function pointers) that are safe or unsafe for whatever reason,
/// with or without said unsafety being explicitely ignored

#[repr(C)]
pub struct SafeStruct (pub i32);

#[repr(C)]
pub struct UnsafeStruct (pub String);

#[repr(C)]
//#[allow(improper_ctype_definitions)]
pub struct AllowedUnsafeStruct (pub String);

// refs are only unsafe if the value comes from the other side of the FFI boundary
// due to the non-null assumption
// (technically there are also assumptions about non-dandling, alignment, aliasing,
//  lifetimes, etc...)
#[repr(C)]
pub struct UnsafeFromForeignStruct<'a> (pub &'a u32);

#[repr(C)]
//#[allow(improper_ctype_definitions)]
pub struct AllowedUnsafeFromForeignStruct<'a> (pub &'a u32);


pub type SafeFnPtr = extern "C" fn(i32)->i32;

pub type UnsafeFnPtr = extern "C" fn((i32,i32))->i32;

#[allow(improper_c_callbacks)]
pub type AllowedUnsafeFnPtr = extern "C" fn(&[i32])->i32;

pub type UnsafeRustCalleeFnPtr = extern "C" fn(i32)->&'static i32;

#[allow(improper_c_callbacks)]
pub type AllowedUnsafeRustCalleeFnPtr = extern "C" fn(i32)->&'static i32;

pub type UnsafeForeignCalleeFnPtr = extern "C" fn(&i32);

#[allow(improper_c_callbacks)]
pub type AllowedUnsafeForeignCalleeFnPtr = extern "C" fn(&i32);


// ////////////////////////////////////
/// types used in specific issue-based tests that need extern-crate types

#[repr(C)]
#[non_exhaustive]
pub struct NonExhaustiveStruct {
    pub field: u8
}

extern "C" {
    pub fn nonexhaustivestruct_create() -> *mut NonExhaustiveStruct;
    pub fn nonexhaustivestruct_destroy(s: *mut NonExhaustiveStruct);
}
