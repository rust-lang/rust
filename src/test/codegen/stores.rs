// compile-flags: -C no-prepopulate-passes
//

#![crate_type = "lib"]

pub struct Bytes {
  a: u8,
  b: u8,
  c: u8,
  d: u8,
}

// CHECK-LABEL: small_array_alignment
#[no_mangle]
pub fn small_array_alignment(x: &mut [i8; 4], y: [i8; 4]) {
    // CHECK: [[TMP:%.+]] = alloca [4 x i8], align 1
    // CHECK: %y = alloca [4 x i8], align 1
    // CHECK: store [4 x i8] %0, [4 x i8]* %y
    // CHECK: [[Y8:%[0-9]+]] = bitcast [4 x i8]* %y to i8*
    // CHECK: [[TMP8:%[0-9]+]] = bitcast [4 x i8]* [[TMP]] to i8*
    *x = y;
}

// CHECK-LABEL: small_struct_alignment
#[no_mangle]
pub fn small_struct_alignment(x: &mut Bytes, y: Bytes) {
    // CHECK: [[TMP:%.+]] = alloca %Bytes, align 1
    // CHECK: %y = alloca %Bytes, align 1
    // CHECK: store %Bytes %0, %Bytes* %y
    // CHECK: [[Y8:%[0-9]+]] = bitcast %Bytes* %y to i8*
    // CHECK: [[TMP8:%[0-9]+]] = bitcast %Bytes* [[TMP]] to i8*
    *x = y;
}
