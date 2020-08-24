// compile-flags: -O

#![crate_type = "lib"]
#![feature(array_map)]

// CHECK-LABEL: @array_inc
// CHECK-NOT: alloca
#[no_mangle]
pub fn array_inc(x: [u8; 1000]) -> [u8; 1000] {
  x.map(|v| v + 1)
}

// CHECK-LABEL: @array_inc_cast
// CHECK: alloca
#[no_mangle]
pub fn array_inc_cast(x: [u8; 1000]) -> [u16; 1000] {
  x.map(|v| v as u16 + 1)
}
