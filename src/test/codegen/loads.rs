// compile-flags: -C no-prepopulate-passes -Zmir-opt-level=0

#![crate_type = "lib"]

pub struct Bytes {
    a: u8,
    b: u8,
    c: u8,
    d: u8,
}

// CHECK: @ptr_alignment_helper({{.*}}align [[PTR_ALIGNMENT:[0-9]+]]
#[no_mangle]
pub fn ptr_alignment_helper(x: &&()) {}

// CHECK-LABEL: @load_raw_pointer
#[no_mangle]
pub fn load_raw_pointer<'a>(x: &*const i32) -> *const i32 {
    // loaded raw pointer should not have !nonnull, !align, or !noundef metadata
    // CHECK: load {{i32\*|ptr}}, {{i32\*\*|ptr}} %x, align [[PTR_ALIGNMENT]]{{$}}
    *x
}

// CHECK-LABEL: @borrow
#[no_mangle]
pub fn borrow(x: &i32) -> &i32 {
    // CHECK: load {{i32\*|ptr}}, {{i32\*\*|ptr}} %x{{.*}}, !nonnull
    &x; // keep variable in an alloca
    x
}

// CHECK-LABEL: @_box
#[no_mangle]
pub fn _box(x: Box<i32>) -> i32 {
    // CHECK: load {{i32\*|ptr}}, {{i32\*\*|ptr}} %x{{.*}}, align [[PTR_ALIGNMENT]]
    *x
}

// CHECK-LABEL: small_array_alignment
// The array is loaded as i32, but its alignment is lower, go with 1 byte to avoid target
// dependent alignment
#[no_mangle]
pub fn small_array_alignment(x: [i8; 4]) -> [i8; 4] {
    // CHECK: [[VAR:%[0-9]+]] = load i32, {{i32\*|ptr}} %{{.*}}, align 1
    // CHECK: ret i32 [[VAR]]
    x
}

// CHECK-LABEL: small_struct_alignment
// The struct is loaded as i32, but its alignment is lower, go with 1 byte to avoid target
// dependent alignment
#[no_mangle]
pub fn small_struct_alignment(x: Bytes) -> Bytes {
    // CHECK: [[VAR:%[0-9]+]] = load i32, {{i32\*|ptr}} %{{.*}}, align 1
    // CHECK: ret i32 [[VAR]]
    x
}
