#![feature(rustc_private)]
#![feature(extern_types)]
#![feature(pattern_types, rustc_attrs)]
#![feature(pattern_type_macro)]
#![feature(unsafe_binders)]
#![allow(private_interfaces)]
#![deny(improper_ctypes, improper_ctypes_definitions)]

use std::cell::UnsafeCell;
use std::ffi::{c_int, c_uint};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::pat::pattern_type;

unsafe extern "C" {
    type UnsizedFFIOpaque;
}
trait Bar {}
trait Mirror {
    type It: ?Sized;
}
impl<T: ?Sized> Mirror for T {
    type It = Self;
}
#[repr(C)]
pub struct StructWithProjection(*mut <StructWithProjection as Mirror>::It);
#[repr(C)]
pub struct StructWithProjectionAndLifetime<'a>(
    &'a mut <StructWithProjectionAndLifetime<'a> as Mirror>::It,
);
pub type I32Pair = (i32, i32);
pub type RustFn = fn();
pub type RustBoxRet = extern "C" fn() -> Box<u32>;
pub type CVoidRet = ();
pub struct Foo;
#[repr(transparent)]
pub struct TransparentI128(i128);
#[repr(transparent)]
pub struct TransparentStr(&'static str);
#[repr(transparent)]
pub struct TransparentBoxFn(RustBoxRet);
#[repr(transparent)]
pub struct TransparentInt(u32);
#[repr(transparent)]
pub struct TransparentRef<'a>(&'a TransparentInt);
#[repr(transparent)]
pub struct TransparentLifetime<'a>(*const u8, ::std::marker::PhantomData<&'a ()>);

#[repr(C)]
pub struct TwoBadTypes<'a> {
    non_c_type: char,
    ref_with_mdata: &'a [u8],
}

#[repr(C)]
pub struct UnsizedStructBecauseForeign {
    sized: u32,
    unszd: UnsizedFFIOpaque,
}
#[repr(C)]
pub struct UnsizedStructBecauseDyn {
    sized: u32,
    unszd: dyn Debug,
}

extern "C" {
    pub fn ptr_type1(size: *const Foo);
    pub fn ptr_type2(size: *const Foo);
    pub fn ptr_unit(p: *const ());
    pub fn ptr_tuple(p: *const ((),));
    pub fn slice_type(p: &[u32]); //~ ERROR: uses type `&[u32]`
    pub fn str_type(p: &str); //~ ERROR: uses type `&str`
    pub fn box_type(p: Box<u32>);
    pub fn opt_box_type(p: Option<Box<u32>>);
    pub fn bool_type(p: bool);
    pub fn char_type(p: char); //~ ERROR uses type `char`
    pub fn pat_type1() -> Option<pattern_type!(u32 is 0..)>; //~ ERROR uses type `Option<pattern_type!(u32 is 0..)>`
    pub fn pat_type2(p: Option<pattern_type!(u32 is 1..)>); // no error!
    pub fn trait_type(p: &dyn Bar); //~ ERROR uses type `&dyn Bar`
    pub fn tuple_type(p: (i32, i32)); //~ ERROR uses type `(i32, i32)`
    pub fn tuple_type2(p: I32Pair); //~ ERROR uses type `(i32, i32)`
    pub fn tuple_unsized(p: Box<(i32, [u8])>); //~ ERROR uses type `Box<(i32, [u8])>`
    pub fn unsafe_binder() -> unsafe<'made_up> &'made_up (i32, UnsizedFFIOpaque);
    //~^ ERROR: uses type `unsafe<'a> &'a (i32, UnsizedFFIOpaque)`

    pub fn zero_size_phantom_toplevel() -> ::std::marker::PhantomData<bool>; //~ ERROR uses type `PhantomData<bool>`
    pub fn fn_type(p: RustFn); //~ ERROR uses type `fn()`
    pub fn fn_type2(p: fn()); //~ ERROR uses type `fn()`
    pub fn fn_contained(p: RustBoxRet);
    pub fn transparent_str(p: TransparentStr); //~ ERROR: uses type `TransparentStr`
    pub fn transparent_fn(p: TransparentBoxFn);
    pub fn raw_array(arr: [u8; 8]); //~ ERROR: uses type `[u8; 8]`

    pub fn multi_errors_per_arg(
        f: for<'a> extern "C" fn(a: char, b: &dyn Debug, c: TwoBadTypes<'a>),
    );
    //~^^ ERROR: uses type `char`
    //~| ERROR: uses type `&dyn Debug`
    //~| ERROR: uses type `TwoBadTypes<'_>`
    //~| ERROR: uses type `TwoBadTypes<'_>`

    pub fn struct_unsized_ptr_no_metadata(p: *const UnsizedStructBecauseForeign);
    pub fn struct_unsized_ptr_has_metadata(p: *const UnsizedStructBecauseDyn); //~ ERROR uses type `*const UnsizedStructBecauseDyn`

    pub fn no_niche_a(a: Option<UnsafeCell<extern "C" fn()>>);
    //~^ ERROR: uses type `Option<UnsafeCell<extern "C" fn()>>`
    pub fn no_niche_b(b: Option<UnsafeCell<&i32>>);
    //~^ ERROR: uses type `Option<UnsafeCell<&i32>>`

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
    #[allow(improper_ctypes)]
    pub fn good18(_: &String);
    pub fn good20(arr: *const [u8; 8]);
    pub static good21: [u8; 8];
    pub fn good_i128_type(p: i128);
    pub fn good_u128_type(p: u128);
    pub fn good_transparent_i128(p: TransparentI128);
    pub static good_static_u128_type: u128;
    pub static good_static_u128_array_type: [u128; 16];

    // note: to have metasized types as arguments without indirection,
    // we need function pointers (don't ask me how it's accepted there)
    pub fn for_fnptr1(f: extern "C" fn(dyn Bar)); //~ ERROR: uses type `dyn Bar`
    pub fn for_fnptr2(f: extern "C" fn(str)); //~ ERROR: uses type `str`
    pub fn for_fnptr3(f: extern "C" fn([u32])); //~ ERROR: uses type `[u32]`
    pub fn for_fnptr4(f: extern "C" fn(UnsizedFFIOpaque));
}

#[allow(improper_ctypes)]
extern "C" {
    pub fn good19(_: &String);
}

static DEFAULT_U32: u32 = 42;
#[no_mangle]
static EXPORTED_STATIC: &u32 = &DEFAULT_U32;
#[no_mangle]
static EXPORTED_STATIC_BAD: &'static str = "is this reaching you, plugin?";
//~^ ERROR: uses type `&str`
#[export_name = "EXPORTED_STATIC_MUT_BUT_RENAMED"]
static mut EXPORTED_STATIC_MUT: &u32 = &DEFAULT_U32;

#[cfg(not(target_arch = "wasm32"))]
extern "C" {
    pub fn good1(size: *const c_int);
    pub fn good2(size: *const c_uint);
}

fn main() {}
