// ignore-tidy-linelength
#![allow(unused, unnecessary_transmutes)]
#![feature(ptr_metadata)]

use std::{ptr, mem};

// Strip out raw byte dumps to make comparison platform-independent:
//@ normalize-stderr: "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr: "([0-9a-f][0-9a-f] |╾─*ALLOC[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"
//@ normalize-stderr: "offset \d+" -> "offset N"
//@ normalize-stderr: "size \d+" -> "size N"
//@ dont-require-annotations: NOTE

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

// # str
// OK
const STR_VALID: &str = unsafe { mem::transmute((&42u8, 1usize)) };
// bad str
const STR_TOO_LONG: &str = unsafe { mem::transmute((&42u8, 999usize)) };
//~^ ERROR encountered a dangling reference (going beyond the bounds of its allocation)
const NESTED_STR_MUCH_TOO_LONG: (&str,) = (unsafe { mem::transmute((&42, usize::MAX)) },);
//~^ ERROR slice is bigger than largest supported object
// bad str
const STR_LENGTH_PTR: &str = unsafe { mem::transmute((&42u8, &3)) };
//~^ ERROR unable to turn pointer into integer
// bad str in user-defined unsized type
const MY_STR_LENGTH_PTR: &MyStr = unsafe { mem::transmute((&42u8, &3)) };
//~^ ERROR unable to turn pointer into integer
const MY_STR_MUCH_TOO_LONG: &MyStr = unsafe { mem::transmute((&42u8, usize::MAX)) };
//~^ ERROR slice is bigger than largest supported object

// uninitialized byte
const STR_NO_INIT: &str = unsafe { mem::transmute::<&[_], _>(&[MaybeUninit::<u8> { uninit: () }]) };
//~^ ERROR uninitialized memory, but expected a string
// uninitialized byte in user-defined str-like
const MYSTR_NO_INIT: &MyStr = unsafe { mem::transmute::<&[_], _>(&[MaybeUninit::<u8> { uninit: () }]) };
//~^ ERROR uninitialized memory, but expected a string

// # slice
// OK
const SLICE_VALID: &[u8] = unsafe { mem::transmute((&42u8, 1usize)) };
// bad slice: length uninit
const SLICE_LENGTH_UNINIT: &[u8] = unsafe {
//~^ ERROR uninitialized
    let uninit_len = MaybeUninit::<usize> { uninit: () };
    mem::transmute((42, uninit_len))
};
// bad slice: length too big
const SLICE_TOO_LONG: &[u8] = unsafe { mem::transmute((&42u8, 999usize)) };
//~^ ERROR dangling reference (going beyond the bounds of its allocation)
// bad slice: length computation overflows
const SLICE_TOO_LONG_OVERFLOW: &[u32] = unsafe { mem::transmute((&42u32, isize::MAX)) };
//~^ ERROR slice is bigger than largest supported object
// bad slice: length not an int
const SLICE_LENGTH_PTR: &[u8] = unsafe { mem::transmute((&42u8, &3)) };
//~^ ERROR unable to turn pointer into integer
// bad slice box: length too big
const SLICE_TOO_LONG_BOX: Box<[u8]> = unsafe { mem::transmute((&42u8, 999usize)) };
//~^ ERROR dangling box (going beyond the bounds of its allocation)
// bad slice box: length not an int
const SLICE_LENGTH_PTR_BOX: Box<[u8]> = unsafe { mem::transmute((&42u8, &3)) };
//~^ ERROR unable to turn pointer into integer

// bad data *inside* the slice
const SLICE_CONTENT_INVALID: &[bool] = &[unsafe { mem::transmute(3u8) }];
//~^ ERROR 0x03, but expected a boolean

// good MySliceBool
const MYSLICE_GOOD: &MySliceBool = &MySlice(true, [false]);
// bad: sized field is not okay
const MYSLICE_PREFIX_BAD: &MySliceBool = &MySlice(unsafe { mem::transmute(3u8) }, [false]);
//~^ ERROR 0x03, but expected a boolean
// bad: unsized part is not okay
const MYSLICE_SUFFIX_BAD: &MySliceBool = &MySlice(true, [unsafe { mem::transmute(3u8) }]);
//~^ ERROR 0x03, but expected a boolean

// # raw slice
const RAW_SLICE_VALID: *const [u8] = unsafe { mem::transmute((&42u8, 1usize)) }; // ok
const RAW_SLICE_TOO_LONG: *const [u8] = unsafe { mem::transmute((&42u8, 999usize)) }; // ok because raw
const RAW_SLICE_MUCH_TOO_LONG: *const [u8] = unsafe { mem::transmute((&42u8, usize::MAX)) }; // ok because raw
const RAW_SLICE_LENGTH_UNINIT: *const [u8] = unsafe {
//~^ ERROR uninitialized
    let uninit_len = MaybeUninit::<usize> { uninit: () };
    mem::transmute((42, uninit_len))
};

// # trait object
// bad trait object
const TRAIT_OBJ_SHORT_VTABLE_1: W<&dyn Trait> = unsafe { mem::transmute(W((&92u8, &3u8))) };
//~^ ERROR vtable
// bad trait object
const TRAIT_OBJ_SHORT_VTABLE_2: W<&dyn Trait> = unsafe { mem::transmute(W((&92u8, &3u64))) };
//~^ ERROR vtable
// bad trait object
const TRAIT_OBJ_INT_VTABLE: W<&dyn Trait> = unsafe { mem::transmute(W((&92u8, 4usize))) };
//~^ ERROR vtable
const TRAIT_OBJ_UNALIGNED_VTABLE: &dyn Trait = unsafe { mem::transmute((&92u8, &[0u8; 128])) };
//~^ ERROR vtable
const TRAIT_OBJ_BAD_DROP_FN_NULL: &dyn Trait = unsafe { mem::transmute((&92u8, &[0usize; 8])) };
//~^ ERROR vtable
const TRAIT_OBJ_BAD_DROP_FN_INT: &dyn Trait = unsafe { mem::transmute((&92u8, &[1usize; 8])) };
//~^ ERROR vtable
const TRAIT_OBJ_BAD_DROP_FN_NOT_FN_PTR: W<&dyn Trait> = unsafe { mem::transmute(W((&92u8, &[&42u8; 8]))) };
//~^ ERROR vtable

// bad data *inside* the trait object
const TRAIT_OBJ_CONTENT_INVALID: &dyn Trait = unsafe { mem::transmute::<_, &bool>(&3u8) };
//~^ ERROR expected a boolean

// # raw trait object
const RAW_TRAIT_OBJ_VTABLE_NULL: *const dyn Trait = unsafe { mem::transmute((&92u8, 0usize)) };
//~^ ERROR null pointer
const RAW_TRAIT_OBJ_VTABLE_INVALID: *const dyn Trait = unsafe { mem::transmute((&92u8, &3u64)) };
//~^ ERROR vtable
const RAW_TRAIT_OBJ_CONTENT_INVALID: *const dyn Trait = unsafe { mem::transmute::<_, &bool>(&3u8) } as *const dyn Trait; // ok because raw
// Officially blessed way to get the vtable
const DYN_METADATA: ptr::DynMetadata<dyn Send> = ptr::metadata::<dyn Send>(ptr::null::<i32>());


static mut RAW_TRAIT_OBJ_VTABLE_NULL_THROUGH_REF: *const dyn Trait = unsafe {
    mem::transmute::<_, &dyn Trait>((&92u8, 0usize))
    //~^^ ERROR null pointer
};
static mut RAW_TRAIT_OBJ_VTABLE_INVALID_THROUGH_REF: *const dyn Trait = unsafe {
    mem::transmute::<_, &dyn Trait>((&92u8, &3u64))
    //~^^ ERROR vtable
};

fn main() {}
