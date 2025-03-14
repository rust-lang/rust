//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

const LUT: [u8; 2] = [1, 1];

// CHECK-LABEL: @decode
#[no_mangle]
pub fn decode(i: u8) -> u8 {
    // CHECK: start:
    // CHECK-NEXT: icmp
    // CHECK-NEXT: select
    // CHECK-NEXT: ret
    if i < 2 { LUT[i as usize] } else { 2 }
}
