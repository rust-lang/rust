#![deny(improper_ctypes)]
#![feature(rustc_private)]

#![allow(private_in_public)]

extern crate libc;

use std::marker::PhantomData;

trait Mirror { type It: ?Sized; }
impl<T: ?Sized> Mirror for T { type It = Self; }
#[repr(C)]
pub struct StructWithProjection(*mut <StructWithProjection as Mirror>::It);
#[repr(C)]
pub struct StructWithProjectionAndLifetime<'a>(
    &'a mut <StructWithProjectionAndLifetime<'a> as Mirror>::It
);
pub type I32Pair = (i32, i32);
#[repr(C)]
pub struct ZeroSize;
pub type RustFn = fn();
pub type RustBadRet = extern fn() -> Box<u32>;
pub type CVoidRet = ();
pub struct Foo;
#[repr(transparent)]
pub struct TransparentI128(i128);
#[repr(transparent)]
pub struct TransparentStr(&'static str);
#[repr(transparent)]
pub struct TransparentBadFn(RustBadRet);
#[repr(transparent)]
pub struct TransparentInt(u32);
#[repr(transparent)]
pub struct TransparentRef<'a>(&'a TransparentInt);
#[repr(transparent)]
pub struct TransparentLifetime<'a>(*const u8, PhantomData<&'a ()>);
#[repr(transparent)]
pub struct TransparentUnit<U>(f32, PhantomData<U>);
#[repr(transparent)]
pub struct TransparentCustomZst(i32, ZeroSize);

#[repr(C)]
pub struct ZeroSizeWithPhantomData(::std::marker::PhantomData<i32>);

extern {
    pub fn ptr_type1(size: *const Foo); //~ ERROR: uses type `Foo`
    pub fn ptr_type2(size: *const Foo); //~ ERROR: uses type `Foo`
    pub fn slice_type(p: &[u32]); //~ ERROR: uses type `[u32]`
    pub fn str_type(p: &str); //~ ERROR: uses type `str`
    pub fn box_type(p: Box<u32>); //~ ERROR uses type `std::boxed::Box<u32>`
    pub fn char_type(p: char); //~ ERROR uses type `char`
    pub fn i128_type(p: i128); //~ ERROR uses type `i128`
    pub fn u128_type(p: u128); //~ ERROR uses type `u128`
    pub fn trait_type(p: &dyn Clone); //~ ERROR uses type `dyn std::clone::Clone`
    pub fn tuple_type(p: (i32, i32)); //~ ERROR uses type `(i32, i32)`
    pub fn tuple_type2(p: I32Pair); //~ ERROR uses type `(i32, i32)`
    pub fn zero_size(p: ZeroSize); //~ ERROR struct has no fields
    pub fn zero_size_phantom(p: ZeroSizeWithPhantomData); //~ ERROR composed only of PhantomData
    pub fn zero_size_phantom_toplevel()
        -> ::std::marker::PhantomData<bool>; //~ ERROR: composed only of PhantomData
    pub fn fn_type(p: RustFn); //~ ERROR function pointer has Rust-specific
    pub fn fn_type2(p: fn()); //~ ERROR function pointer has Rust-specific
    pub fn fn_contained(p: RustBadRet); //~ ERROR: uses type `std::boxed::Box<u32>`
    pub fn transparent_i128(p: TransparentI128); //~ ERROR: uses type `i128`
    pub fn transparent_str(p: TransparentStr); //~ ERROR: uses type `str`
    pub fn transparent_fn(p: TransparentBadFn); //~ ERROR: uses type `std::boxed::Box<u32>`

    pub fn good3(fptr: Option<extern fn()>);
    pub fn good4(aptr: &[u8; 4 as usize]);
    pub fn good5(s: StructWithProjection);
    pub fn good6(s: StructWithProjectionAndLifetime);
    pub fn good7(fptr: extern fn() -> ());
    pub fn good8(fptr: extern fn() -> !);
    pub fn good9() -> ();
    pub fn good10() -> CVoidRet;
    pub fn good11(size: isize);
    pub fn good12(size: usize);
    pub fn good13(n: TransparentInt);
    pub fn good14(p: TransparentRef);
    pub fn good15(p: TransparentLifetime);
    pub fn good16(p: TransparentUnit<ZeroSize>);
    pub fn good17(p: TransparentCustomZst);
    #[allow(improper_ctypes)]
    pub fn good18(_: &String);
}

#[allow(improper_ctypes)]
extern {
    pub fn good19(_: &String);
}

#[cfg(not(target_arch = "wasm32"))]
extern {
    pub fn good1(size: *const libc::c_int);
    pub fn good2(size: *const libc::c_uint);
}

fn main() {
}
