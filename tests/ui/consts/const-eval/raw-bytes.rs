//@ stderr-per-bitwidth
//@ ignore-endian-big
// ignore-tidy-linelength
//@ normalize-stderr: "╾─*ALLOC[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼" -> "╾ALLOC_ID$1╼"
//@ dont-require-annotations: NOTE

#![allow(invalid_value, unnecessary_transmutes)]
#![feature(never_type, rustc_attrs, ptr_metadata, slice_from_ptr_range, const_slice_from_ptr_range)]

use std::mem;
use std::alloc::Layout;
use std::ptr::NonNull;
use std::num::NonZero;
use std::slice::{from_ptr_range, from_raw_parts};

// # Bad enums and chars

#[repr(usize)]
#[derive(Copy, Clone)]
enum Enum {
    A = 0,
}
const BAD_ENUM: Enum = unsafe { mem::transmute(1usize) };
//~^ ERROR constructing invalid value

#[repr(usize)]
#[derive(Copy, Clone)]
enum Enum2 {
    A = 2,
}
const BAD_ENUM2: Enum2 = unsafe { mem::transmute(0usize) };
//~^ ERROR constructing invalid value

#[derive(Copy, Clone)]
enum Never {}

// An enum with 3 variants of which some are uninhabited -- so the uninhabited variants *do*
// have a discriminant.
enum UninhDiscriminant {
    A,
    B(!),
    C,
    D(Never),
}
const BAD_UNINHABITED_VARIANT1: UninhDiscriminant = unsafe { mem::transmute(1u8) };
//~^ ERROR constructing invalid value
const BAD_UNINHABITED_VARIANT2: UninhDiscriminant = unsafe { mem::transmute(3u8) };
//~^ ERROR constructing invalid value

// Invalid enum field content (mostly to test printing of paths for enum tuple
// variants and tuples).
// Need to create something which does not clash with enum layout optimizations.
const BAD_OPTION_CHAR: Option<(char, char)> = Some(('x', unsafe { mem::transmute(!0u32) }));
//~^ ERROR constructing invalid value

// # Bad pointers and references

const NULL_PTR: NonNull<u8> = unsafe { mem::transmute(0usize) };
//~^ ERROR constructing invalid value

const NULL_U8: NonZero<u8> = unsafe { mem::transmute(0u8) };
//~^ ERROR constructing invalid value
const NULL_USIZE: NonZero<usize> = unsafe { mem::transmute(0usize) };
//~^ ERROR constructing invalid value

#[rustc_layout_scalar_valid_range_start(10)]
#[rustc_layout_scalar_valid_range_end(30)]
struct RestrictedRange1(u32);
const BAD_RANGE1: RestrictedRange1 = unsafe { RestrictedRange1(42) };
//~^ ERROR constructing invalid value

#[rustc_layout_scalar_valid_range_start(30)]
#[rustc_layout_scalar_valid_range_end(10)]
struct RestrictedRange2(u32);
const BAD_RANGE2: RestrictedRange2 = unsafe { RestrictedRange2(20) };
//~^ ERROR constructing invalid value

const NULL_FAT_PTR: NonNull<dyn Send> = unsafe {
    //~^ ERROR constructing invalid value
    let x: &dyn Send = &42;
    let meta = std::ptr::metadata(x);
    mem::transmute((0_usize, meta))
};

const UNALIGNED: &u16 = unsafe { mem::transmute(&[0u8; 4]) };
//~^ ERROR constructing invalid value: encountered an unaligned reference (required 2 byte alignment but found 1)

const UNALIGNED_BOX: Box<u16> = unsafe { mem::transmute(&[0u8; 4]) };
//~^ ERROR constructing invalid value: encountered an unaligned box (required 2 byte alignment but found 1)

const NULL: &u16 = unsafe { mem::transmute(0usize) };
//~^ ERROR constructing invalid value

const NULL_BOX: Box<u16> = unsafe { mem::transmute(0usize) };
//~^ ERROR constructing invalid value

const USIZE_AS_REF: &'static u8 = unsafe { mem::transmute(1337usize) };
//~^ ERROR constructing invalid value

const USIZE_AS_BOX: Box<u8> = unsafe { mem::transmute(1337usize) };
//~^ ERROR constructing invalid value

const NULL_FN_PTR: fn() = unsafe { mem::transmute(0usize) };
//~^ ERROR constructing invalid value
const DANGLING_FN_PTR: fn() = unsafe { mem::transmute(13usize) };
//~^ ERROR constructing invalid value
const DATA_FN_PTR: fn() = unsafe { mem::transmute(&13) };
//~^ ERROR constructing invalid value

#[derive(Copy, Clone)]
enum Bar {}

const BAD_BAD_REF: &Bar = unsafe { mem::transmute(1usize) };
//~^ ERROR constructing invalid value

/// A newtype wrapper to prevent MIR generation from inserting reborrows that would affect the error
/// message.
#[repr(transparent)]
struct W<T>(T);

#[repr(C)]
union MaybeUninit<T: Copy> {
    uninit: (),
    init: T,
}

trait Trait {}
impl Trait for bool {}

// custom unsized type
struct MyStr(str);

// custom unsized type with sized fields
struct MySlice<T: ?Sized>(bool, T);
type MySliceBool = MySlice<[bool]>;

const STR_TOO_LONG: &str = unsafe { mem::transmute((&42u8, 999usize)) };
//~^ ERROR constructing invalid value
const NESTED_STR_MUCH_TOO_LONG: (&str,) = (unsafe { mem::transmute((&42, usize::MAX)) },);
//~^ ERROR constructing invalid value
const MY_STR_MUCH_TOO_LONG: &MyStr = unsafe { mem::transmute((&42u8, usize::MAX)) };
//~^ ERROR constructing invalid value

const STR_NO_INIT: &str = unsafe { mem::transmute::<&[_], _>(&[MaybeUninit::<u8> { uninit: () }]) };
//~^ ERROR constructing invalid value
const MYSTR_NO_INIT: &MyStr = unsafe { mem::transmute::<&[_], _>(&[MaybeUninit::<u8> { uninit: () }]) };
//~^ ERROR constructing invalid value
const MYSTR_NO_INIT_ISSUE83182: &MyStr = unsafe { mem::transmute::<&[_], _>(&[&()]) };
//~^ ERROR: constructing invalid value

// # slice
const SLICE_TOO_LONG: &[u8] = unsafe { mem::transmute((&42u8, 999usize)) };
//~^ ERROR constructing invalid value
const SLICE_TOO_LONG_OVERFLOW: &[u32] = unsafe { mem::transmute((&42u32, isize::MAX)) };
//~^ ERROR constructing invalid value
// bad slice box: length too big
const SLICE_TOO_LONG_BOX: Box<[u8]> = unsafe { mem::transmute((&42u8, 999usize)) };
//~^ ERROR constructing invalid value
// bad data *inside* the slice
const SLICE_CONTENT_INVALID: &[bool] = &[unsafe { mem::transmute(3u8) }];
//~^ ERROR encountered 0x03, but expected a boolean

// bad: sized field is not okay
const MYSLICE_PREFIX_BAD: &MySliceBool = &MySlice(unsafe { mem::transmute(3u8) }, [false]);
//~^ ERROR encountered 0x03, but expected a boolean
// bad: unsized part is not okay
const MYSLICE_SUFFIX_BAD: &MySliceBool = &MySlice(true, [unsafe { mem::transmute(3u8) }]);
//~^ ERROR encountered 0x03, but expected a boolean

// bad trait object
const TRAIT_OBJ_SHORT_VTABLE_1: W<&dyn Trait> = unsafe { mem::transmute(W((&92u8, &3u8))) };
//~^ ERROR expected a vtable
// bad trait object
const TRAIT_OBJ_SHORT_VTABLE_2: W<&dyn Trait> = unsafe { mem::transmute(W((&92u8, &3u64))) };
//~^ ERROR expected a vtable
// bad trait object
const TRAIT_OBJ_INT_VTABLE: W<&dyn Trait> = unsafe { mem::transmute(W((&92u8, 4usize))) };
//~^ ERROR expected a vtable
const TRAIT_OBJ_BAD_DROP_FN_NOT_FN_PTR: W<&dyn Trait> = unsafe { mem::transmute(W((&92u8, &[&42u8; 8]))) };
//~^ ERROR expected a vtable
// bad data *inside* the trait object
const TRAIT_OBJ_CONTENT_INVALID: &dyn Trait = unsafe { mem::transmute::<_, &bool>(&3u8) };
//~^ ERROR expected a boolean

const RAW_TRAIT_OBJ_VTABLE_NULL: *const dyn Trait = unsafe { mem::transmute((&92u8, 0usize)) };
//~^ ERROR null pointer
const RAW_TRAIT_OBJ_VTABLE_INVALID: *const dyn Trait = unsafe { mem::transmute((&92u8, &3u64)) };
//~^ ERROR vtable

// Uninhabited types
const _: &[!; 1] = unsafe { &*(1_usize as *const [!; 1]) }; //~ ERROR constructing invalid value
const _: &[!] = unsafe { &*(1_usize as *const [!; 1]) }; //~ ERROR constructing invalid value
const _: &[!] = unsafe { &*(1_usize as *const [!; 42]) }; //~ ERROR constructing invalid value

// Reading uninitialized  data
pub static S4: &[u8] = unsafe { from_raw_parts((&D1) as *const _ as _, 1) };
//~^ ERROR: constructing invalid value
// Reinterpret pointers as integers (UB in CTFE.)
pub static S5: &[u8] = unsafe { from_raw_parts((&D3) as *const _ as _, mem::size_of::<&u32>()) };
//~^ ERROR: constructing invalid value
// Layout mismatch
pub static S6: &[bool] = unsafe { from_raw_parts((&D0) as *const _ as _, 4) };
//~^ ERROR: constructing invalid value

// Reading padding is not ok
pub static S7: &[u16] = unsafe {
    //~^ ERROR: constructing invalid value
    let ptr = (&D2 as *const Struct as *const u16).add(1);

    from_raw_parts(ptr, 4)
};

pub static R4: &[u8] = unsafe {
    //~^ ERROR: constructing invalid value
    let ptr = (&D1) as *const mem::MaybeUninit<&u32> as *const u8;
    from_ptr_range(ptr..ptr.add(1))
};
pub static R5: &[u8] = unsafe {
    //~^ ERROR: constructing invalid value
    let ptr = &D3 as *const &u32;
    from_ptr_range(ptr.cast()..ptr.add(1).cast())
};
pub static R6: &[bool] = unsafe {
    //~^ ERROR: constructing invalid value
    let ptr = &D0 as *const u32 as *const bool;
    from_ptr_range(ptr..ptr.add(4))
};

const D0: u32 = 0x11111111; // Constant chosen for endianness-independent behavior.
const D1: mem::MaybeUninit<&u32> = mem::MaybeUninit::uninit();
const D2: Struct = Struct { a: 1, b: 2, c: 3, d: 4 };
const D3: &u32 = &42;

#[repr(C)]
struct Struct {
    a: u8,
    // _pad: [mem::MaybeUninit<u8>; 3]
    b: u32,
    c: u16,
    d: u8,
    // _pad: [mem::MaybeUninit<u8>; 1]
}

fn main() {}
