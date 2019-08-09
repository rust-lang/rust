#![allow(unused)]
#![allow(const_err)] // make sure we cannot allow away the errors tested here

// normalize-stderr-test "alignment \d+" -> "alignment N"
// normalize-stderr-test "offset \d+" -> "offset N"
// normalize-stderr-test "allocation \d+" -> "allocation N"
// normalize-stderr-test "size \d+" -> "size N"

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

union SliceTransmute {
    repr: SliceRepr,
    bad: BadSliceRepr,
    slice: &'static [u8],
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

union DynTransmute {
    repr: DynRepr,
    repr2: DynRepr2,
    bad: BadDynRepr,
    rust: &'static dyn Trait,
}

trait Trait {}
impl Trait for bool {}

// custom unsized type
struct MyStr(str);

// custom unsized type with sized fields
struct MySlice<T: ?Sized>(bool, T);
type MySliceBool = MySlice<[bool]>;

// OK
const A: &str = unsafe { SliceTransmute { repr: SliceRepr { ptr: &42, len: 1 } }.str};
// bad str
const B: &str = unsafe { SliceTransmute { repr: SliceRepr { ptr: &42, len: 999 } }.str};
//~^ ERROR it is undefined behavior to use this value
// bad str
const C: &str = unsafe { SliceTransmute { bad: BadSliceRepr { ptr: &42, len: &3 } }.str};
//~^ ERROR it is undefined behavior to use this value
// bad str in user-defined unsized type
const C2: &MyStr = unsafe { SliceTransmute { bad: BadSliceRepr { ptr: &42, len: &3 } }.my_str};
//~^ ERROR it is undefined behavior to use this value

// OK
const A2: &[u8] = unsafe { SliceTransmute { repr: SliceRepr { ptr: &42, len: 1 } }.slice};
// bad slice
const B2: &[u8] = unsafe { SliceTransmute { repr: SliceRepr { ptr: &42, len: 999 } }.slice};
//~^ ERROR it is undefined behavior to use this value
// bad slice
const C3: &[u8] = unsafe { SliceTransmute { bad: BadSliceRepr { ptr: &42, len: &3 } }.slice};
//~^ ERROR it is undefined behavior to use this value

// bad trait object
const D: &dyn Trait = unsafe { DynTransmute { repr: DynRepr { ptr: &92, vtable: &3 } }.rust};
//~^ ERROR it is undefined behavior to use this value
// bad trait object
const E: &dyn Trait = unsafe { DynTransmute { repr2: DynRepr2 { ptr: &92, vtable: &3 } }.rust};
//~^ ERROR it is undefined behavior to use this value
// bad trait object
const F: &dyn Trait = unsafe { DynTransmute { bad: BadDynRepr { ptr: &92, vtable: 3 } }.rust};
//~^ ERROR it is undefined behavior to use this value

// bad data *inside* the trait object
const G: &dyn Trait = &unsafe { BoolTransmute { val: 3 }.bl };
//~^ ERROR it is undefined behavior to use this value

// bad data *inside* the slice
const H: &[bool] = &[unsafe { BoolTransmute { val: 3 }.bl }];
//~^ ERROR it is undefined behavior to use this value

// good MySliceBool
const I1: &MySliceBool = &MySlice(true, [false]);
// bad: sized field is not okay
const I2: &MySliceBool = &MySlice(unsafe { BoolTransmute { val: 3 }.bl }, [false]);
//~^ ERROR it is undefined behavior to use this value
// bad: unsized part is not okay
const I3: &MySliceBool = &MySlice(true, [unsafe { BoolTransmute { val: 3 }.bl }]);
//~^ ERROR it is undefined behavior to use this value

// invalid UTF-8
const J1: &str = unsafe { SliceTransmute { slice: &[0xFF] }.str };
//~^ ERROR it is undefined behavior to use this value
// invalid UTF-8 in user-defined str-like
const J2: &MyStr = unsafe { SliceTransmute { slice: &[0xFF] }.my_str };
//~^ ERROR it is undefined behavior to use this value

fn main() {
}
