// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

#[no_mangle]
pub struct F32(f32);

// CHECK: define{{.*}}float @add_newtype_f32(float noundef %a, float noundef %b)
#[inline(never)]
#[no_mangle]
pub fn add_newtype_f32(a: F32, b: F32) -> F32 {
    F32(a.0 + b.0)
}

#[no_mangle]
pub struct F64(f64);

// CHECK: define{{.*}}double @add_newtype_f64(double noundef %a, double noundef %b)
#[inline(never)]
#[no_mangle]
pub fn add_newtype_f64(a: F64, b: F64) -> F64 {
    F64(a.0 + b.0)
}
