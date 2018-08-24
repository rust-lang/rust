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
    my_str: &'static Str,
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
    rust: &'static Trait,
}

trait Trait {}
impl Trait for bool {}

struct Str(str);

// OK
const A: &str = unsafe { SliceTransmute { repr: SliceRepr { ptr: &42, len: 1 } }.str};
// bad str
const B: &str = unsafe { SliceTransmute { repr: SliceRepr { ptr: &42, len: 999 } }.str};
//~^ ERROR this constant likely exhibits undefined behavior
// bad str
const C: &str = unsafe { SliceTransmute { bad: BadSliceRepr { ptr: &42, len: &3 } }.str};
//~^ ERROR this constant likely exhibits undefined behavior
// bad str in Str
const C2: &Str = unsafe { SliceTransmute { bad: BadSliceRepr { ptr: &42, len: &3 } }.my_str};
//~^ ERROR this constant likely exhibits undefined behavior

// OK
const A2: &[u8] = unsafe { SliceTransmute { repr: SliceRepr { ptr: &42, len: 1 } }.slice};
// bad slice
const B2: &[u8] = unsafe { SliceTransmute { repr: SliceRepr { ptr: &42, len: 999 } }.slice};
//~^ ERROR this constant likely exhibits undefined behavior
// bad slice
const C3: &[u8] = unsafe { SliceTransmute { bad: BadSliceRepr { ptr: &42, len: &3 } }.slice};
//~^ ERROR this constant likely exhibits undefined behavior

// bad trait object
const D: &Trait = unsafe { DynTransmute { repr: DynRepr { ptr: &92, vtable: &3 } }.rust};
//~^ ERROR this constant likely exhibits undefined behavior
// bad trait object
const E: &Trait = unsafe { DynTransmute { repr2: DynRepr2 { ptr: &92, vtable: &3 } }.rust};
//~^ ERROR this constant likely exhibits undefined behavior
// bad trait object
const F: &Trait = unsafe { DynTransmute { bad: BadDynRepr { ptr: &92, vtable: 3 } }.rust};
//~^ ERROR this constant likely exhibits undefined behavior

// bad data *inside* the trait object
const G: &Trait = &unsafe { BoolTransmute { val: 3 }.bl };
//~^ ERROR this constant likely exhibits undefined behavior

// bad data *inside* the slice
const H: &[bool] = &[unsafe { BoolTransmute { val: 3 }.bl }];
//~^ ERROR this constant likely exhibits undefined behavior

fn main() {
}
