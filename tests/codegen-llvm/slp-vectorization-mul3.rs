//! Regression test for #142519
//@ only-x86_64
//@ compile-flags: -O
//@ min-llvm-version: 22

#![crate_type = "lib"]

// CHECK-LABEL: @mul3
// CHECK: phi <4 x i8>
// CHECK: load <4 x i8>
// CHECK: add <4 x i8>
// CHECK: store <4 x i8>

#[no_mangle]
pub fn mul3(previous: &[[u8; 4]], current: &mut [[u8; 4]]) {
    let mut c_bpp = [0u8; 4];

    for i in 0..previous.len() {
        current[i][0] = current[i][0].wrapping_add(c_bpp[0]);
        current[i][1] = current[i][1].wrapping_add(c_bpp[1]);
        current[i][2] = current[i][2].wrapping_add(c_bpp[2]);
        current[i][3] = current[i][3].wrapping_add(c_bpp[3]);

        c_bpp = previous[i];
    }
}
