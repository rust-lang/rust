// ignore-tidy-linelength
#![allow(unused)]
#![allow(const_err)] // make sure we cannot allow away the errors tested here

// normalize-stderr-test "offset \d+" -> "offset N"
// normalize-stderr-test "allocation \d+" -> "allocation N"
// normalize-stderr-test "size \d+" -> "size N"

#[repr(C)]
union BoolTransmute {
  val: u8,
  bl: bool,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct SliceRepr {
    ptr: *const u8,
    len: usize,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct BadSliceRepr {
    ptr: *const u8,
    len: &'static u8,
}

#[repr(C)]
union SliceTransmute {
    repr: SliceRepr,
    bad: BadSliceRepr,
    addr: usize,
    slice: &'static [u8],
    raw_slice: *const [u8],
    str: &'static str,
    my_str: &'static MyStr,
    my_slice: &'static MySliceBool,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct DynRepr {
    ptr: *const u8,
    vtable: *const u8,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct DynRepr2 {
    ptr: *const u8,
    vtable: *const u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct BadDynRepr {
    ptr: *const u8,
    vtable: usize,
}

#[repr(C)]
union DynTransmute {
    repr: DynRepr,
    repr2: DynRepr2,
    bad: BadDynRepr,
    addr: usize,
    rust: &'static dyn Trait,
    raw_rust: *const dyn Trait,
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
const STR_VALID: &str = unsafe { SliceTransmute { repr: SliceRepr { ptr: &42, len: 1 } }.str};
// bad str
const STR_TOO_LONG: &str = unsafe { SliceTransmute { repr: SliceRepr { ptr: &42, len: 999 } }.str};
//~^ ERROR it is undefined behavior to use this value
// bad str
const STR_LENGTH_PTR: &str = unsafe { SliceTransmute { bad: BadSliceRepr { ptr: &42, len: &3 } }.str};
//~^ ERROR it is undefined behavior to use this value
// bad str in user-defined unsized type
const MY_STR_LENGTH_PTR: &MyStr = unsafe { SliceTransmute { bad: BadSliceRepr { ptr: &42, len: &3 } }.my_str};
//~^ ERROR it is undefined behavior to use this value

// invalid UTF-8
const STR_NO_UTF8: &str = unsafe { SliceTransmute { slice: &[0xFF] }.str };
//~^ ERROR it is undefined behavior to use this value
// invalid UTF-8 in user-defined str-like
const MYSTR_NO_UTF8: &MyStr = unsafe { SliceTransmute { slice: &[0xFF] }.my_str };
//~^ ERROR it is undefined behavior to use this value

// # slice
// OK
const SLICE_VALID: &[u8] = unsafe { SliceTransmute { repr: SliceRepr { ptr: &42, len: 1 } }.slice};
// bad slice: length uninit
const SLICE_LENGTH_UNINIT: &[u8] = unsafe { SliceTransmute { addr: 42 }.slice};
//~^ ERROR it is undefined behavior to use this value
// bad slice: length too big
const SLICE_TOO_LONG: &[u8] = unsafe { SliceTransmute { repr: SliceRepr { ptr: &42, len: 999 } }.slice};
//~^ ERROR it is undefined behavior to use this value
// bad slice: length not an int
const SLICE_LENGTH_PTR: &[u8] = unsafe { SliceTransmute { bad: BadSliceRepr { ptr: &42, len: &3 } }.slice};
//~^ ERROR it is undefined behavior to use this value

// bad data *inside* the slice
const SLICE_CONTENT_INVALID: &[bool] = &[unsafe { BoolTransmute { val: 3 }.bl }];
//~^ ERROR it is undefined behavior to use this value

// good MySliceBool
const MYSLICE_GOOD: &MySliceBool = &MySlice(true, [false]);
// bad: sized field is not okay
const MYSLICE_PREFIX_BAD: &MySliceBool = &MySlice(unsafe { BoolTransmute { val: 3 }.bl }, [false]);
//~^ ERROR it is undefined behavior to use this value
// bad: unsized part is not okay
const MYSLICE_SUFFIX_BAD: &MySliceBool = &MySlice(true, [unsafe { BoolTransmute { val: 3 }.bl }]);
//~^ ERROR it is undefined behavior to use this value

// # raw slice
const RAW_SLICE_VALID: *const [u8] = unsafe { SliceTransmute { repr: SliceRepr { ptr: &42, len: 1 } }.raw_slice}; // ok
const RAW_SLICE_TOO_LONG: *const [u8] = unsafe { SliceTransmute { repr: SliceRepr { ptr: &42, len: 999 } }.raw_slice}; // ok because raw
const RAW_SLICE_MUCH_TOO_LONG: *const [u8] = unsafe { SliceTransmute { repr: SliceRepr { ptr: &42, len: usize::max_value() } }.raw_slice}; // ok because raw
const RAW_SLICE_LENGTH_UNINIT: *const [u8] = unsafe { SliceTransmute { addr: 42 }.raw_slice};
//~^ ERROR it is undefined behavior to use this value

// # trait object
// bad trait object
const TRAIT_OBJ_SHORT_VTABLE_1: &dyn Trait = unsafe { DynTransmute { repr: DynRepr { ptr: &92, vtable: &3 } }.rust};
//~^ ERROR it is undefined behavior to use this value
// bad trait object
const TRAIT_OBJ_SHORT_VTABLE_2: &dyn Trait = unsafe { DynTransmute { repr2: DynRepr2 { ptr: &92, vtable: &3 } }.rust};
//~^ ERROR it is undefined behavior to use this value
// bad trait object
const TRAIT_OBJ_INT_VTABLE: &dyn Trait = unsafe { DynTransmute { bad: BadDynRepr { ptr: &92, vtable: 3 } }.rust};
//~^ ERROR it is undefined behavior to use this value

// bad data *inside* the trait object
const TRAIT_OBJ_CONTENT_INVALID: &dyn Trait = &unsafe { BoolTransmute { val: 3 }.bl };
//~^ ERROR it is undefined behavior to use this value

// # raw trait object
const RAW_TRAIT_OBJ_VTABLE_NULL: *const dyn Trait = unsafe { DynTransmute { bad: BadDynRepr { ptr: &92, vtable: 0 } }.raw_rust};
//~^ ERROR it is undefined behavior to use this value
const RAW_TRAIT_OBJ_VTABLE_INVALID: *const dyn Trait = unsafe { DynTransmute { repr2: DynRepr2 { ptr: &92, vtable: &3 } }.raw_rust};
//~^ ERROR it is undefined behavior to use this value
const RAW_TRAIT_OBJ_CONTENT_INVALID: *const dyn Trait = &unsafe { BoolTransmute { val: 3 }.bl } as *const _; // ok because raw

// Const eval fails for these, so they need to be statics to error.
static mut RAW_TRAIT_OBJ_VTABLE_NULL_THROUGH_REF: *const dyn Trait = unsafe {
    DynTransmute { bad: BadDynRepr { ptr: &92, vtable: 0 } }.rust
    //~^ ERROR could not evaluate static initializer
};
static mut RAW_TRAIT_OBJ_VTABLE_INVALID_THROUGH_REF: *const dyn Trait = unsafe {
    DynTransmute { repr2: DynRepr2 { ptr: &92, vtable: &3 } }.rust
    //~^ ERROR could not evaluate static initializer
};

fn main() {
    let _ = RAW_TRAIT_OBJ_VTABLE_NULL;
    let _ = RAW_TRAIT_OBJ_VTABLE_INVALID;
}
