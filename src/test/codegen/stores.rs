// compile-flags: -C no-prepopulate-passes
// ignore-tidy-linelength
// min-llvm-version 7.0

#![crate_type = "lib"]

pub struct Bytes {
  a: u8,
  b: u8,
  c: u8,
  d: u8,
}

// CHECK-LABEL: small_array_alignment
// The array is stored as i32, but its alignment is lower, go with 1 byte to avoid target
// dependent alignment
#[no_mangle]
pub fn small_array_alignment(x: &mut [i8; 4], y: [i8; 4]) {
// CHECK: [[TMP:%.+]] = alloca i32
// CHECK: %y = alloca [4 x i8]
// CHECK: store i32 %0, i32* [[TMP]]
// CHECK: [[Y8:%[0-9]+]] = bitcast [4 x i8]* %y to i8*
// CHECK: [[TMP8:%[0-9]+]] = bitcast i32* [[TMP]] to i8*
// CHECK: call void @llvm.memcpy.{{.*}}(i8* align 1 [[Y8]], i8* align 4 [[TMP8]], i{{[0-9]+}} 4, i1 false)
    *x = y;
}

// CHECK-LABEL: small_struct_alignment
// The struct is stored as i32, but its alignment is lower, go with 1 byte to avoid target
// dependent alignment
#[no_mangle]
pub fn small_struct_alignment(x: &mut Bytes, y: Bytes) {
// CHECK: [[TMP:%.+]] = alloca i32
// CHECK: %y = alloca %Bytes
// CHECK: store i32 %0, i32* [[TMP]]
// CHECK: [[Y8:%[0-9]+]] = bitcast %Bytes* %y to i8*
// CHECK: [[TMP8:%[0-9]+]] = bitcast i32* [[TMP]] to i8*
// CHECK: call void @llvm.memcpy.{{.*}}(i8* align 1 [[Y8]], i8* align 4 [[TMP8]], i{{[0-9]+}} 4, i1 false)
    *x = y;
}
