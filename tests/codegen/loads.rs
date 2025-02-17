//@ compile-flags: -C no-prepopulate-passes -Zmir-opt-level=0 -Copt-level=3

#![crate_type = "lib"]

use std::mem::MaybeUninit;
use std::num::NonZero;

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

#[repr(align(16))]
pub struct Align16(u128);

// CHECK: @ptr_alignment_helper({{.*}}align [[PTR_ALIGNMENT:[0-9]+]]
#[no_mangle]
pub fn ptr_alignment_helper(x: &&()) {}

// CHECK-LABEL: @load_ref
#[no_mangle]
pub fn load_ref<'a>(x: &&'a i32) -> &'a i32 {
    // CHECK: load ptr, ptr %x, align [[PTR_ALIGNMENT]], !nonnull !{{[0-9]+}}, !align ![[ALIGN_4_META:[0-9]+]], !noundef !{{[0-9]+}}
    *x
}

// CHECK-LABEL: @load_ref_higher_alignment
#[no_mangle]
pub fn load_ref_higher_alignment<'a>(x: &&'a Align16) -> &'a Align16 {
    // CHECK: load ptr, ptr %x, align [[PTR_ALIGNMENT]], !nonnull !{{[0-9]+}}, !align ![[ALIGN_16_META:[0-9]+]], !noundef !{{[0-9]+}}
    *x
}

// CHECK-LABEL: @load_scalar_pair
#[no_mangle]
pub fn load_scalar_pair<'a>(x: &(&'a i32, &'a Align16)) -> (&'a i32, &'a Align16) {
    // CHECK: load ptr, ptr %{{.+}}, align [[PTR_ALIGNMENT]], !nonnull !{{[0-9]+}}, !align ![[ALIGN_4_META]], !noundef !{{[0-9]+}}
    // CHECK: load ptr, ptr %{{.+}}, align [[PTR_ALIGNMENT]], !nonnull !{{[0-9]+}}, !align ![[ALIGN_16_META]], !noundef !{{[0-9]+}}
    *x
}

// CHECK-LABEL: @load_raw_pointer
#[no_mangle]
pub fn load_raw_pointer<'a>(x: &*const i32) -> *const i32 {
    // loaded raw pointer should not have !nonnull or !align metadata
    // CHECK: load ptr, ptr %x, align [[PTR_ALIGNMENT]], !noundef ![[NOUNDEF:[0-9]+]]{{$}}
    *x
}

// CHECK-LABEL: @load_box
#[no_mangle]
pub fn load_box<'a>(x: Box<Box<i32>>) -> Box<i32> {
    // CHECK: load ptr, ptr %{{.*}}, align [[PTR_ALIGNMENT]], !nonnull !{{[0-9]+}}, !align ![[ALIGN_4_META]], !noundef !{{[0-9]+}}
    *x
}

// CHECK-LABEL: @load_bool
#[no_mangle]
pub fn load_bool(x: &bool) -> bool {
    // CHECK: load i8, ptr %x, align 1, !range ![[BOOL_RANGE:[0-9]+]], !noundef !{{[0-9]+}}
    *x
}

// CHECK-LABEL: @load_maybeuninit_bool
#[no_mangle]
pub fn load_maybeuninit_bool(x: &MaybeUninit<bool>) -> MaybeUninit<bool> {
    // CHECK: load i8, ptr %x, align 1{{$}}
    *x
}

// CHECK-LABEL: @load_enum_bool
#[no_mangle]
pub fn load_enum_bool(x: &MyBool) -> MyBool {
    // CHECK: load i8, ptr %x, align 1, !range ![[BOOL_RANGE]], !noundef !{{[0-9]+}}
    *x
}

// CHECK-LABEL: @load_maybeuninit_enum_bool
#[no_mangle]
pub fn load_maybeuninit_enum_bool(x: &MaybeUninit<MyBool>) -> MaybeUninit<MyBool> {
    // CHECK: load i8, ptr %x, align 1{{$}}
    *x
}

// CHECK-LABEL: @load_int
#[no_mangle]
pub fn load_int(x: &u16) -> u16 {
    // CHECK: load i16, ptr %x, align 2, !noundef ![[NOUNDEF]]{{$}}
    *x
}

// CHECK-LABEL: @load_nonzero_int
#[no_mangle]
pub fn load_nonzero_int(x: &NonZero<u16>) -> NonZero<u16> {
    // CHECK: load i16, ptr %x, align 2, !range ![[NONZEROU16_RANGE:[0-9]+]], !noundef !{{[0-9]+}}
    *x
}

// CHECK-LABEL: @load_option_nonzero_int
#[no_mangle]
pub fn load_option_nonzero_int(x: &Option<NonZero<u16>>) -> Option<NonZero<u16>> {
    // CHECK: load i16, ptr %x, align 2, !noundef ![[NOUNDEF]]{{$}}
    *x
}

// CHECK-LABEL: @borrow
#[no_mangle]
pub fn borrow(x: &i32) -> &i32 {
    // CHECK: load ptr, ptr %x{{.*}}, !nonnull
    &x; // keep variable in an alloca
    x
}

// CHECK-LABEL: @_box
#[no_mangle]
pub fn _box(x: Box<i32>) -> i32 {
    // CHECK: load ptr, ptr %x{{.*}}, align [[PTR_ALIGNMENT]]
    *x
}

// CHECK-LABEL: small_array_alignment
// The array is loaded as i32, but its alignment is lower, go with 1 byte to avoid target
// dependent alignment
#[no_mangle]
pub fn small_array_alignment(x: [i8; 4]) -> [i8; 4] {
    // CHECK: [[VAR:%[0-9]+]] = load i32, ptr %{{.*}}, align 1
    // CHECK: ret i32 [[VAR]]
    x
}

// CHECK-LABEL: small_struct_alignment
// The struct is loaded as i32, but its alignment is lower, go with 1 byte to avoid target
// dependent alignment
#[no_mangle]
pub fn small_struct_alignment(x: Bytes) -> Bytes {
    // CHECK: [[VAR:%[0-9]+]] = load i32, ptr %{{.*}}, align 1
    // CHECK: ret i32 [[VAR]]
    x
}

// CHECK-DAG: ![[BOOL_RANGE]] = !{i8 0, i8 2}
// CHECK-DAG: ![[NONZEROU16_RANGE]] = !{i16 1, i16 0}
// CHECK-DAG: ![[ALIGN_4_META]] = !{i64 4}
// CHECK-DAG: ![[ALIGN_16_META]] = !{i64 16}
