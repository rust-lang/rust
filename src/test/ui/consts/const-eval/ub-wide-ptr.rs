// ignore-tidy-linelength
#![allow(unused)]
#![allow(const_err)] // make sure we cannot allow away the errors tested here

use std::mem;

// normalize-stderr-test "offset \d+" -> "offset N"
// normalize-stderr-test "alloc\d+" -> "allocN"
// normalize-stderr-test "size \d+" -> "size N"

/// A newtype wrapper to prevent MIR generation from inserting reborrows that would affect the error
/// message. Use this whenever the message is "any use of this value will cause an error" instead of
/// "it is undefined behavior to use this value".
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
//~^ ERROR it is undefined behavior to use this value
const NESTED_STR_MUCH_TOO_LONG: (&str,) = (unsafe { mem::transmute((&42, usize::MAX)) },);
//~^ ERROR it is undefined behavior to use this value
// bad str
const STR_LENGTH_PTR: &str = unsafe { mem::transmute((&42u8, &3)) };
//~^ ERROR it is undefined behavior to use this value
// bad str in user-defined unsized type
const MY_STR_LENGTH_PTR: &MyStr = unsafe { mem::transmute((&42u8, &3)) };
//~^ ERROR it is undefined behavior to use this value
const MY_STR_MUCH_TOO_LONG: &MyStr = unsafe { mem::transmute((&42u8, usize::MAX)) };
//~^ ERROR it is undefined behavior to use this value

// uninitialized byte
const STR_NO_INIT: &str = unsafe { mem::transmute::<&[_], _>(&[MaybeUninit::<u8> { uninit: () }]) };
//~^ ERROR it is undefined behavior to use this value
// uninitialized byte in user-defined str-like
const MYSTR_NO_INIT: &MyStr = unsafe { mem::transmute::<&[_], _>(&[MaybeUninit::<u8> { uninit: () }]) };
//~^ ERROR it is undefined behavior to use this value

// # slice
// OK
const SLICE_VALID: &[u8] = unsafe { mem::transmute((&42u8, 1usize)) };
// bad slice: length uninit
const SLICE_LENGTH_UNINIT: &[u8] = unsafe {
//~^ ERROR it is undefined behavior to use this value
    let uninit_len = MaybeUninit::<usize> { uninit: () };
    mem::transmute((42, uninit_len))
};
// bad slice: length too big
const SLICE_TOO_LONG: &[u8] = unsafe { mem::transmute((&42u8, 999usize)) };
//~^ ERROR it is undefined behavior to use this value
// bad slice: length not an int
const SLICE_LENGTH_PTR: &[u8] = unsafe { mem::transmute((&42u8, &3)) };
//~^ ERROR it is undefined behavior to use this value
// bad slice box: length too big
const SLICE_TOO_LONG_BOX: Box<[u8]> = unsafe { mem::transmute((&42u8, 999usize)) };
//~^ ERROR it is undefined behavior to use this value
// bad slice box: length not an int
const SLICE_LENGTH_PTR_BOX: Box<[u8]> = unsafe { mem::transmute((&42u8, &3)) };
//~^ ERROR it is undefined behavior to use this value

// bad data *inside* the slice
const SLICE_CONTENT_INVALID: &[bool] = &[unsafe { mem::transmute(3u8) }];
//~^ ERROR it is undefined behavior to use this value

// good MySliceBool
const MYSLICE_GOOD: &MySliceBool = &MySlice(true, [false]);
// bad: sized field is not okay
const MYSLICE_PREFIX_BAD: &MySliceBool = &MySlice(unsafe { mem::transmute(3u8) }, [false]);
//~^ ERROR it is undefined behavior to use this value
// bad: unsized part is not okay
const MYSLICE_SUFFIX_BAD: &MySliceBool = &MySlice(true, [unsafe { mem::transmute(3u8) }]);
//~^ ERROR it is undefined behavior to use this value

// # raw slice
const RAW_SLICE_VALID: *const [u8] = unsafe { mem::transmute((&42u8, 1usize)) }; // ok
const RAW_SLICE_TOO_LONG: *const [u8] = unsafe { mem::transmute((&42u8, 999usize)) }; // ok because raw
const RAW_SLICE_MUCH_TOO_LONG: *const [u8] = unsafe { mem::transmute((&42u8, usize::MAX)) }; // ok because raw
const RAW_SLICE_LENGTH_UNINIT: *const [u8] = unsafe {
//~^ ERROR it is undefined behavior to use this value
    let uninit_len = MaybeUninit::<usize> { uninit: () };
    mem::transmute((42, uninit_len))
};

// # trait object
// bad trait object
const TRAIT_OBJ_SHORT_VTABLE_1: W<&dyn Trait> = unsafe { mem::transmute(W((&92u8, &3u8))) };
//~^ ERROR it is undefined behavior to use this value
// bad trait object
const TRAIT_OBJ_SHORT_VTABLE_2: W<&dyn Trait> = unsafe { mem::transmute(W((&92u8, &3u64))) };
//~^ ERROR it is undefined behavior to use this value
// bad trait object
const TRAIT_OBJ_INT_VTABLE: W<&dyn Trait> = unsafe { mem::transmute(W((&92u8, 4usize))) };
//~^ ERROR it is undefined behavior to use this value
const TRAIT_OBJ_UNALIGNED_VTABLE: &dyn Trait = unsafe { mem::transmute((&92u8, &[0u8; 128])) };
//~^ ERROR it is undefined behavior to use this value
const TRAIT_OBJ_BAD_DROP_FN_NULL: &dyn Trait = unsafe { mem::transmute((&92u8, &[0usize; 8])) };
//~^ ERROR it is undefined behavior to use this value
const TRAIT_OBJ_BAD_DROP_FN_INT: &dyn Trait = unsafe { mem::transmute((&92u8, &[1usize; 8])) };
//~^ ERROR it is undefined behavior to use this value
const TRAIT_OBJ_BAD_DROP_FN_NOT_FN_PTR: W<&dyn Trait> = unsafe { mem::transmute(W((&92u8, &[&42u8; 8]))) };
//~^ ERROR it is undefined behavior to use this value

// bad data *inside* the trait object
const TRAIT_OBJ_CONTENT_INVALID: &dyn Trait = unsafe { mem::transmute::<_, &bool>(&3u8) };
//~^ ERROR it is undefined behavior to use this value

// # raw trait object
const RAW_TRAIT_OBJ_VTABLE_NULL: *const dyn Trait = unsafe { mem::transmute((&92u8, 0usize)) };
//~^ ERROR it is undefined behavior to use this value
const RAW_TRAIT_OBJ_VTABLE_INVALID: *const dyn Trait = unsafe { mem::transmute((&92u8, &3u64)) };
//~^ ERROR it is undefined behavior to use this value
const RAW_TRAIT_OBJ_CONTENT_INVALID: *const dyn Trait = unsafe { mem::transmute::<_, &bool>(&3u8) } as *const dyn Trait; // ok because raw

// Const eval fails for these, so they need to be statics to error.
static mut RAW_TRAIT_OBJ_VTABLE_NULL_THROUGH_REF: *const dyn Trait = unsafe {
    mem::transmute::<_, &dyn Trait>((&92u8, 0usize))
    //~^ ERROR could not evaluate static initializer
};
static mut RAW_TRAIT_OBJ_VTABLE_INVALID_THROUGH_REF: *const dyn Trait = unsafe {
    mem::transmute::<_, &dyn Trait>((&92u8, &3u64))
    //~^ ERROR could not evaluate static initializer
};

fn main() {}
