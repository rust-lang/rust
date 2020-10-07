// compile-flags: -C opt-level=3 -Zmir-opt-level=3
#![crate_type = "lib"]
#![feature(array_map)]

const SIZE: usize = 4;

// CHECK-LABEL: @array_cast_to_float
#[no_mangle]
pub fn array_cast_to_float(x: [u32; SIZE]) -> [f32; SIZE] {
  // CHECK: cast
  // CHECK: @llvm.memcpy
  // CHECK: ret
  // CHECK-EMPTY
  x.map(|v| v as f32)
}

// CHECK-LABEL: @array_cast_to_u64
#[no_mangle]
pub fn array_cast_to_u64(x: [u32; SIZE]) -> [u64; SIZE] {
  // CHECK: cast
  // CHECK: @llvm.memcpy
  // CHECK: ret
  // CHECK-EMPTY
  x.map(|v| v as u64)
}
