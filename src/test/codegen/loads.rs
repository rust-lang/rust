// compile-flags: -C no-prepopulate-passes -Zmir-opt-level=0

#![crate_type = "lib"]

use std::mem::MaybeUninit;
use std::num::NonZeroU16;

pub struct Bytes {
    a: u8,
    b: u8,
    c: u8,
    d: u8,
}

#[derive(Copy, Clone)]
pub enum MyBool {
    True,
    False,
}

// CHECK-LABEL: @load_ref
#[no_mangle]
pub fn load_ref<'a>(x: &&'a i32) -> &'a i32 {
// Alignment of a reference itself is target dependent, so just match any alignment:
// the main thing we care about here is !nonnull and !noundef.
// CHECK: load i32*, i32** %x, align {{[0-9]+}}, !nonnull !{{[0-9]+}}, !noundef !{{[0-9]+}}
    *x
}

// CHECK-LABEL: @load_box
#[no_mangle]
pub fn load_box<'a>(x: Box<Box<i32>>) -> Box<i32> {
// Alignment of a box itself is target dependent, so just match any alignment:
// the main thing we care about here is !nonnull and !noundef.
// CHECK: load i32*, i32** %x, align {{[0-9]+}}, !nonnull !{{[0-9]+}}, !noundef !{{[0-9]+}}
    *x
}

// CHECK-LABEL: @load_bool
#[no_mangle]
pub fn load_bool(x: &bool) -> bool {
// CHECK: load i8, i8* %x, align 1, !range ![[BOOL_RANGE:[0-9]+]], !noundef !{{[0-9]+}}
    *x
}

// CHECK-LABEL: @load_maybeuninit_bool
#[no_mangle]
pub fn load_maybeuninit_bool(x: &MaybeUninit<bool>) -> MaybeUninit<bool> {
// CHECK: load i8, i8* %x, align 1{{$}}
    *x
}

// CHECK-LABEL: @load_enum_bool
#[no_mangle]
pub fn load_enum_bool(x: &MyBool) -> MyBool {
// CHECK: load i8, i8* %x, align 1, !range ![[BOOL_RANGE]], !noundef !{{[0-9]+}}
    *x
}

// CHECK-LABEL: @load_maybeuninit_enum_bool
#[no_mangle]
pub fn load_maybeuninit_enum_bool(x: &MaybeUninit<MyBool>) -> MaybeUninit<MyBool> {
// CHECK: load i8, i8* %x, align 1{{$}}
    *x
}

// CHECK-LABEL: @load_int
#[no_mangle]
pub fn load_int(x: &u16) -> u16 {
// CHECK: load i16, i16* %x, align 2{{$}}
    *x
}

// CHECK-LABEL: @load_nonzero_int
#[no_mangle]
pub fn load_nonzero_int(x: &NonZeroU16) -> NonZeroU16 {
// CHECK: load i16, i16* %x, align 2, !range ![[NONZEROU16_RANGE:[0-9]+]], !noundef !{{[0-9]+}}
    *x
}

// CHECK-LABEL: @load_option_nonzero_int
#[no_mangle]
pub fn load_option_nonzero_int(x: &Option<NonZeroU16>) -> Option<NonZeroU16> {
// CHECK: load i16, i16* %x, align 2{{$}}
    *x
}

// CHECK-LABEL: @borrow
#[no_mangle]
pub fn borrow(x: &i32) -> &i32 {
// CHECK: load {{(i32\*, )?}}i32** %x{{.*}}, !nonnull
    &x; // keep variable in an alloca
    x
}

// CHECK-LABEL: @_box
#[no_mangle]
pub fn _box(x: Box<i32>) -> i32 {
// CHECK: load {{(i32\*, )?}}i32** %x{{.*}}, !nonnull
    *x
}

// CHECK-LABEL: small_array_alignment
// The array is loaded as i32, but its alignment is lower, go with 1 byte to avoid target
// dependent alignment
#[no_mangle]
pub fn small_array_alignment(x: [i8; 4]) -> [i8; 4] {
// CHECK: [[VAR:%[0-9]+]] = load {{(i32, )?}}i32* %{{.*}}, align 1
// CHECK: ret i32 [[VAR]]
    x
}

// CHECK-LABEL: small_struct_alignment
// The struct is loaded as i32, but its alignment is lower, go with 1 byte to avoid target
// dependent alignment
#[no_mangle]
pub fn small_struct_alignment(x: Bytes) -> Bytes {
// CHECK: [[VAR:%[0-9]+]] = load {{(i32, )?}}i32* %{{.*}}, align 1
// CHECK: ret i32 [[VAR]]
    x
}

// CHECK: ![[BOOL_RANGE]] = !{i8 0, i8 2}
// CHECK: ![[NONZEROU16_RANGE]] = !{i16 1, i16 0}
