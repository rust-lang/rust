//! Regression test for #142519
//@ only-x86_64
//@ compile-flags: -O

#![crate_type = "lib"]

// CHECK-LABEL: mul3
// CHECK: %[[C_BPP:.*]] = phi <4 x i8>
// CHECK: %[[V:.*]] = load <4 x i8>, ptr
// CHECK: %[[ADD:.*]] = add <4 x i8> %[[V]], %[[C_BPP]]
// CHECK: store {{<4 x i8>|i32}} {{.*}}, ptr

pub fn mul3(previous: &[u8], current: &mut [u8]) {
    let mut c_bpp = [0u8; 4];

    for (chunk, b_bpp) in current.chunks_exact_mut(4).zip(previous.chunks_exact(4)) {
        let new_chunk = [
            chunk[0].wrapping_add(c_bpp[0]),
            chunk[1].wrapping_add(c_bpp[1]),
            chunk[2].wrapping_add(c_bpp[2]),
            chunk[3].wrapping_add(c_bpp[3]),
        ];
        chunk.copy_from_slice(&new_chunk);
        c_bpp.copy_from_slice(b_bpp);
    }
}
