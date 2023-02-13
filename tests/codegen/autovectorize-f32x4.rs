// compile-flags: -C opt-level=3 -Z merge-functions=disabled
// only-x86_64
#![crate_type = "lib"]
#![feature(array_zip)]

// CHECK-LABEL: @auto_vectorize_direct
#[no_mangle]
pub fn auto_vectorize_direct(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
// CHECK: load <4 x float>
// CHECK: load <4 x float>
// CHECK: fadd <4 x float>
// CHECK: store <4 x float>
    [
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
    ]
}

// CHECK-LABEL: @auto_vectorize_loop
#[no_mangle]
pub fn auto_vectorize_loop(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
// CHECK: load <4 x float>
// CHECK: load <4 x float>
// CHECK: fadd <4 x float>
// CHECK: store <4 x float>
    let mut c = [0.0; 4];
    for i in 0..4 {
        c[i] = a[i] + b[i];
    }
    c
}

// CHECK-LABEL: @auto_vectorize_array_zip_map
#[no_mangle]
pub fn auto_vectorize_array_zip_map(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
// CHECK: load <4 x float>
// CHECK: load <4 x float>
// CHECK: fadd <4 x float>
// CHECK: store <4 x float>
    a.zip(b).map(|(a, b)| a + b)
}
