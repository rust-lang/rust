// only-wasm32
// compile-flags: -C target-feature=+nontrapping-fptoint
#![crate_type = "lib"]

// CHECK-LABEL: @cast_f64_i64
#[no_mangle]
pub fn cast_f64_i64(a: f64) -> i64 {
    // CHECK: tail call i64 @llvm.wasm.trunc.saturate.signed.i64.f64(double {{.*}})
    // CHECK-NEXT: ret i64 {{.*}}
    a as _
}

// CHECK-LABEL: @cast_f64_i32
#[no_mangle]
pub fn cast_f64_i32(a: f64) -> i32 {
    // CHECK: tail call i32 @llvm.wasm.trunc.saturate.signed.i32.f64(double {{.*}})
    // CHECK-NEXT: ret i32 {{.*}}
    a as _
}

// CHECK-LABEL: @cast_f32_i64
#[no_mangle]
pub fn cast_f32_i64(a: f32) -> i64 {
    // CHECK: tail call i64 @llvm.wasm.trunc.saturate.signed.i64.f32(float {{.*}})
    // CHECK-NEXT: ret i64 {{.*}}
    a as _
}

// CHECK-LABEL: @cast_f32_i32
#[no_mangle]
pub fn cast_f32_i32(a: f32) -> i32 {
    // CHECK: tail call i32 @llvm.wasm.trunc.saturate.signed.i32.f32(float {{.*}})
    // CHECK-NEXT: ret i32 {{.*}}
    a as _
}


// CHECK-LABEL: @cast_f64_u64
#[no_mangle]
pub fn cast_f64_u64(a: f64) -> u64 {
    // CHECK: tail call i64 @llvm.wasm.trunc.saturate.unsigned.i64.f64(double {{.*}})
    // CHECK-NEXT: ret i64 {{.*}}
    a as _
}

// CHECK-LABEL: @cast_f64_u32
#[no_mangle]
pub fn cast_f64_u32(a: f64) -> u32 {
    // CHECK: tail call i32 @llvm.wasm.trunc.saturate.unsigned.i32.f64(double {{.*}})
    // CHECK-NEXT: ret i32 {{.*}}
    a as _
}

// CHECK-LABEL: @cast_f32_u64
#[no_mangle]
pub fn cast_f32_u64(a: f32) -> u64 {
    // CHECK: tail call i64 @llvm.wasm.trunc.saturate.unsigned.i64.f32(float {{.*}})
    // CHECK-NEXT: ret i64 {{.*}}
    a as _
}

// CHECK-LABEL: @cast_f32_u32
#[no_mangle]
pub fn cast_f32_u32(a: f32) -> u32 {
    // CHECK: tail call i32 @llvm.wasm.trunc.saturate.unsigned.i32.f32(float {{.*}})
    // CHECK-NEXT: ret i32 {{.*}}
    a as _
}

// CHECK-LABEL: @cast_f32_u8
#[no_mangle]
pub fn cast_f32_u8(a: f32) -> u8 {
    // CHECK-NOT: {{.*}} call {{.*}} @llvm.wasm.trunc.{{.*}}
    // CHECK: fptoui float {{.*}} to i8
    // CHECK-NEXT: select i1 {{.*}}, i8 {{.*}}, i8 {{.*}}
    // CHECK-NEXT: ret i8 {{.*}}
    a as _
}



// CHECK-LABEL: @cast_unchecked_f64_i64
#[no_mangle]
pub unsafe fn cast_unchecked_f64_i64(a: f64) -> i64 {
    // CHECK-NOT: {{.*}} call {{.*}} @llvm.wasm.trunc.{{.*}}
    // CHECK: fptosi double {{.*}} to i64
    // CHECK-NEXT: ret i64 {{.*}}
    a.to_int_unchecked()
}

// CHECK-LABEL: @cast_unchecked_f64_i32
#[no_mangle]
pub unsafe fn cast_unchecked_f64_i32(a: f64) -> i32 {
    // CHECK-NOT: {{.*}} call {{.*}} @llvm.wasm.trunc.{{.*}}
    // CHECK: fptosi double {{.*}} to i32
    // CHECK-NEXT: ret i32 {{.*}}
    a.to_int_unchecked()
}

// CHECK-LABEL: @cast_unchecked_f32_i64
#[no_mangle]
pub unsafe fn cast_unchecked_f32_i64(a: f32) -> i64 {
    // CHECK-NOT: {{.*}} call {{.*}} @llvm.wasm.trunc.{{.*}}
    // CHECK: fptosi float {{.*}} to i64
    // CHECK-NEXT: ret i64 {{.*}}
    a.to_int_unchecked()
}

// CHECK-LABEL: @cast_unchecked_f32_i32
#[no_mangle]
pub unsafe fn cast_unchecked_f32_i32(a: f32) -> i32 {
    // CHECK-NOT: {{.*}} call {{.*}} @llvm.wasm.trunc.{{.*}}
    // CHECK: fptosi float {{.*}} to i32
    // CHECK-NEXT: ret i32 {{.*}}
    a.to_int_unchecked()
}


// CHECK-LABEL: @cast_unchecked_f64_u64
#[no_mangle]
pub unsafe fn cast_unchecked_f64_u64(a: f64) -> u64 {
    // CHECK-NOT: {{.*}} call {{.*}} @llvm.wasm.trunc.{{.*}}
    // CHECK: fptoui double {{.*}} to i64
    // CHECK-NEXT: ret i64 {{.*}}
    a.to_int_unchecked()
}

// CHECK-LABEL: @cast_unchecked_f64_u32
#[no_mangle]
pub unsafe fn cast_unchecked_f64_u32(a: f64) -> u32 {
    // CHECK-NOT: {{.*}} call {{.*}} @llvm.wasm.trunc.{{.*}}
    // CHECK: fptoui double {{.*}} to i32
    // CHECK-NEXT: ret i32 {{.*}}
    a.to_int_unchecked()
}

// CHECK-LABEL: @cast_unchecked_f32_u64
#[no_mangle]
pub unsafe fn cast_unchecked_f32_u64(a: f32) -> u64 {
    // CHECK-NOT: {{.*}} call {{.*}} @llvm.wasm.trunc.{{.*}}
    // CHECK: fptoui float {{.*}} to i64
    // CHECK-NEXT: ret i64 {{.*}}
    a.to_int_unchecked()
}

// CHECK-LABEL: @cast_unchecked_f32_u32
#[no_mangle]
pub unsafe fn cast_unchecked_f32_u32(a: f32) -> u32 {
    // CHECK-NOT: {{.*}} call {{.*}} @llvm.wasm.trunc.{{.*}}
    // CHECK: fptoui float {{.*}} to i32
    // CHECK-NEXT: ret i32 {{.*}}
    a.to_int_unchecked()
}

// CHECK-LABEL: @cast_unchecked_f32_u8
#[no_mangle]
pub unsafe fn cast_unchecked_f32_u8(a: f32) -> u8 {
    // CHECK-NOT: {{.*}} call {{.*}} @llvm.wasm.trunc.{{.*}}
    // CHECK: fptoui float {{.*}} to i8
    // CHECK-NEXT: ret i8 {{.*}}
    a.to_int_unchecked()
}
