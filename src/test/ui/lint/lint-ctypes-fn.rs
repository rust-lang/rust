#![feature(rustc_private)]

#![allow(private_in_public)]
#![deny(improper_ctypes_definitions)]

extern crate libc;

use std::default::Default;
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
pub struct ZeroSizeWithPhantomData(PhantomData<i32>);

pub extern "C" fn ptr_type1(size: *const Foo) { }

pub extern "C" fn ptr_type2(size: *const Foo) { }

pub extern "C" fn slice_type(p: &[u32]) { }
//~^ ERROR: uses type `[u32]`

pub extern "C" fn str_type(p: &str) { }
//~^ ERROR: uses type `str`

pub extern "C" fn box_type(p: Box<u32>) { }

pub extern "C" fn opt_box_type(p: Option<Box<u32>>) { }

pub extern "C" fn char_type(p: char) { }
//~^ ERROR uses type `char`

pub extern "C" fn i128_type(p: i128) { }
//~^ ERROR uses type `i128`

pub extern "C" fn u128_type(p: u128) { }
//~^ ERROR uses type `u128`

pub extern "C" fn tuple_type(p: (i32, i32)) { }
//~^ ERROR uses type `(i32, i32)`

pub extern "C" fn tuple_type2(p: I32Pair) { }
//~^ ERROR uses type `(i32, i32)`

pub extern "C" fn zero_size(p: ZeroSize) { }
//~^ ERROR uses type `ZeroSize`

pub extern "C" fn zero_size_phantom(p: ZeroSizeWithPhantomData) { }
//~^ ERROR uses type `ZeroSizeWithPhantomData`

pub extern "C" fn zero_size_phantom_toplevel() -> PhantomData<bool> {
//~^ ERROR uses type `PhantomData<bool>`
    Default::default()
}

pub extern "C" fn fn_type(p: RustFn) { }
//~^ ERROR uses type `fn()`

pub extern "C" fn fn_type2(p: fn()) { }
//~^ ERROR uses type `fn()`

pub extern "C" fn fn_contained(p: RustBadRet) { }

pub extern "C" fn transparent_i128(p: TransparentI128) { }
//~^ ERROR: uses type `i128`

pub extern "C" fn transparent_str(p: TransparentStr) { }
//~^ ERROR: uses type `str`

pub extern "C" fn transparent_fn(p: TransparentBadFn) { }

pub extern "C" fn good3(fptr: Option<extern "C" fn()>) { }

pub extern "C" fn good4(aptr: &[u8; 4 as usize]) { }

pub extern "C" fn good5(s: StructWithProjection) { }

pub extern "C" fn good6(s: StructWithProjectionAndLifetime) { }

pub extern "C" fn good7(fptr: extern "C" fn() -> ()) { }

pub extern "C" fn good8(fptr: extern "C" fn() -> !) { }

pub extern "C" fn good9() -> () { }

pub extern "C" fn good10() -> CVoidRet { }

pub extern "C" fn good11(size: isize) { }

pub extern "C" fn good12(size: usize) { }

pub extern "C" fn good13(n: TransparentInt) { }

pub extern "C" fn good14(p: TransparentRef) { }

pub extern "C" fn good15(p: TransparentLifetime) { }

pub extern "C" fn good16(p: TransparentUnit<ZeroSize>) { }

pub extern "C" fn good17(p: TransparentCustomZst) { }

#[allow(improper_ctypes_definitions)]
pub extern "C" fn good18(_: &String) { }

#[cfg(not(target_arch = "wasm32"))]
pub extern "C" fn good1(size: *const libc::c_int) { }

#[cfg(not(target_arch = "wasm32"))]
pub extern "C" fn good2(size: *const libc::c_uint) { }

pub extern "C" fn unused_generic1<T>(size: *const Foo) { }

pub extern "C" fn unused_generic2<T>() -> PhantomData<bool> {
//~^ ERROR uses type `PhantomData<bool>`
    Default::default()
}

pub extern "C" fn used_generic1<T>(x: T) { }

pub extern "C" fn used_generic2<T>(x: T, size: *const Foo) { }

pub extern "C" fn used_generic3<T: Default>() -> T {
    Default::default()
}

pub extern "C" fn used_generic4<T>(x: Vec<T>) { }
//~^ ERROR: uses type `Vec<T>`

pub extern "C" fn used_generic5<T>() -> Vec<T> {
//~^ ERROR: uses type `Vec<T>`
    Default::default()
}

fn main() {}
