#![feature(rustc_private)]

#![allow(private_in_public)]
#![deny(improper_ctypes)]

extern crate libc;

use std::cell::UnsafeCell;
use std::marker::PhantomData;

trait Bar { }
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
pub type RustBadRet = extern "C" fn() -> Box<u32>;
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

extern "C" {
    pub fn ptr_type1(size: *const Foo); //~ ERROR: uses type `Foo`
    pub fn ptr_type2(size: *const Foo); //~ ERROR: uses type `Foo`
    pub fn ptr_unit(p: *const ());
    pub fn ptr_tuple(p: *const ((),)); //~ ERROR: uses type `((),)`
    pub fn slice_type(p: &[u32]); //~ ERROR: uses type `[u32]`
    pub fn str_type(p: &str); //~ ERROR: uses type `str`
    pub fn box_type(p: Box<u32>); //~ ERROR uses type `Box<u32>`
    pub fn opt_box_type(p: Option<Box<u32>>);
    //~^ ERROR uses type `Option<Box<u32>>`
    pub fn char_type(p: char); //~ ERROR uses type `char`
    pub fn i128_type(p: i128); //~ ERROR uses type `i128`
    pub fn u128_type(p: u128); //~ ERROR uses type `u128`
    pub fn trait_type(p: &dyn Bar); //~ ERROR uses type `dyn Bar`
    pub fn tuple_type(p: (i32, i32)); //~ ERROR uses type `(i32, i32)`
    pub fn tuple_type2(p: I32Pair); //~ ERROR uses type `(i32, i32)`
    pub fn zero_size(p: ZeroSize); //~ ERROR uses type `ZeroSize`
    pub fn zero_size_phantom(p: ZeroSizeWithPhantomData);
    //~^ ERROR uses type `ZeroSizeWithPhantomData`
    pub fn zero_size_phantom_toplevel()
        -> ::std::marker::PhantomData<bool>; //~ ERROR uses type `PhantomData<bool>`
    pub fn fn_type(p: RustFn); //~ ERROR uses type `fn()`
    pub fn fn_type2(p: fn()); //~ ERROR uses type `fn()`
    pub fn fn_contained(p: RustBadRet); //~ ERROR: uses type `Box<u32>`
    pub fn transparent_i128(p: TransparentI128); //~ ERROR: uses type `i128`
    pub fn transparent_str(p: TransparentStr); //~ ERROR: uses type `str`
    pub fn transparent_fn(p: TransparentBadFn); //~ ERROR: uses type `Box<u32>`
    pub fn raw_array(arr: [u8; 8]); //~ ERROR: uses type `[u8; 8]`

    pub fn no_niche_a(a: Option<UnsafeCell<extern fn()>>);
    //~^ ERROR: uses type `Option<UnsafeCell<extern "C" fn()>>`
    pub fn no_niche_b(b: Option<UnsafeCell<&i32>>);
    //~^ ERROR: uses type `Option<UnsafeCell<&i32>>`

    pub static static_u128_type: u128; //~ ERROR: uses type `u128`
    pub static static_u128_array_type: [u128; 16]; //~ ERROR: uses type `u128`

    pub fn good3(fptr: Option<extern "C" fn()>);
    pub fn good4(aptr: &[u8; 4 as usize]);
    pub fn good5(s: StructWithProjection);
    pub fn good6(s: StructWithProjectionAndLifetime);
    pub fn good7(fptr: extern "C" fn() -> ());
    pub fn good8(fptr: extern "C" fn() -> !);
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
    pub fn good20(arr: *const [u8; 8]);
    pub static good21: [u8; 8];

}

#[allow(improper_ctypes)]
extern "C" {
    pub fn good19(_: &String);
}

#[cfg(not(target_arch = "wasm32"))]
extern "C" {
    pub fn good1(size: *const libc::c_int);
    pub fn good2(size: *const libc::c_uint);
}

fn main() {
}
