// compile-flags: -C no-prepopulate-passes -Zmir-opt-level=0

#![crate_type = "lib"]

pub struct Bytes {
  a: u8,
  b: u8,
  c: u8,
  d: u8,
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
#[no_mangle]
pub fn small_array_alignment(x: [i8; 4]) -> [i8; 4] {
    // CHECK: [[VAR:%[0-9]+]] = load [4 x i8], [4 x i8]* %{{.*}}, align 1
    // CHECK: ret [4 x i8] [[VAR]]
    x
}

// CHECK-LABEL: @small_struct_alignment
#[no_mangle]
pub fn small_struct_alignment(x: Bytes) -> Bytes {
    // TODO-CHECK: [[VAR:%[0-9]+]] = load {{(i32, )?}}i32* %{{.*}}, align 1
    // TODO-CHECK: ret i32 [[VAR]]
    x
}
