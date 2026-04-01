//@ only-wasm32
//@ compile-flags: -C target-feature=-nontrapping-fptoint
#![crate_type = "lib"]

// CHECK-LABEL: @cast_f64_i64
#[no_mangle]
pub fn cast_f64_i64(a: f64) -> i64 {
    // CHECK-NOT: fptosi double {{.*}} to i64
    // CHECK-NOT: select i1 {{.*}}, i64 {{.*}}, i64 {{.*}}
    // CHECK: {{.*}} call {{.*}} @llvm.fptosi.sat.i64.f64{{.*}}
    a as _
}

// CHECK-LABEL: @cast_f64_i32
#[no_mangle]
pub fn cast_f64_i32(a: f64) -> i32 {
    // CHECK-NOT: fptosi double {{.*}} to i32
    // CHECK-NOT: select i1 {{.*}}, i32 {{.*}}, i32 {{.*}}
    // CHECK: {{.*}} call {{.*}} @llvm.fptosi.sat.i32.f64{{.*}}
    a as _
}

// CHECK-LABEL: @cast_f32_i64
#[no_mangle]
pub fn cast_f32_i64(a: f32) -> i64 {
    // CHECK-NOT: fptosi float {{.*}} to i64
    // CHECK-NOT: select i1 {{.*}}, i64 {{.*}}, i64 {{.*}}
    // CHECK: {{.*}} call {{.*}} @llvm.fptosi.sat.i64.f32{{.*}}
    a as _
}

// CHECK-LABEL: @cast_f32_i32
#[no_mangle]
pub fn cast_f32_i32(a: f32) -> i32 {
    // CHECK-NOT: fptosi float {{.*}} to i32
    // CHECK-NOT: select i1 {{.*}}, i32 {{.*}}, i32 {{.*}}
    // CHECK: {{.*}} call {{.*}} @llvm.fptosi.sat.i32.f32{{.*}}
    a as _
}

// CHECK-LABEL: @cast_f64_u64
#[no_mangle]
pub fn cast_f64_u64(a: f64) -> u64 {
    // CHECK-NOT: fptoui double {{.*}} to i64
    // CHECK-NOT: select i1 {{.*}}, i64 {{.*}}, i64 {{.*}}
    // CHECK: {{.*}} call {{.*}} @llvm.fptoui.sat.i64.f64{{.*}}
    a as _
}

// CHECK-LABEL: @cast_f64_u32
#[no_mangle]
pub fn cast_f64_u32(a: f64) -> u32 {
    // CHECK-NOT: fptoui double {{.*}} to i32
    // CHECK-NOT: select i1 {{.*}}, i32 {{.*}}, i32 {{.*}}
    // CHECK: {{.*}} call {{.*}} @llvm.fptoui.sat.i32.f64{{.*}}
    a as _
}

// CHECK-LABEL: @cast_f32_u64
#[no_mangle]
pub fn cast_f32_u64(a: f32) -> u64 {
    // CHECK-NOT: fptoui float {{.*}} to i64
    // CHECK-NOT: select i1 {{.*}}, i64 {{.*}}, i64 {{.*}}
    // CHECK: {{.*}} call {{.*}} @llvm.fptoui.sat.i64.f32{{.*}}
    a as _
}

// CHECK-LABEL: @cast_f32_u32
#[no_mangle]
pub fn cast_f32_u32(a: f32) -> u32 {
    // CHECK-NOT: fptoui float {{.*}} to i32
    // CHECK-NOT: select i1 {{.*}}, i32 {{.*}}, i32 {{.*}}
    // CHECK: {{.*}} call {{.*}} @llvm.fptoui.sat.i32.f32{{.*}}
    a as _
}

// CHECK-LABEL: @cast_f32_u8
#[no_mangle]
pub fn cast_f32_u8(a: f32) -> u8 {
    // CHECK-NOT: fptoui float {{.*}} to i8
    // CHECK-NOT: select i1 {{.*}}, i8 {{.*}}, i8 {{.*}}
    // CHECK: {{.*}} call {{.*}} @llvm.fptoui.sat.i8.f32{{.*}}
    a as _
}

// CHECK-LABEL: @cast_unchecked_f64_i64
#[no_mangle]
pub unsafe fn cast_unchecked_f64_i64(a: f64) -> i64 {
    // CHECK: {{.*}} call {{.*}} @llvm.wasm.trunc.signed.{{.*}}
    // CHECK-NEXT: ret i64 {{.*}}
    a.to_int_unchecked()
}

// CHECK-LABEL: @cast_unchecked_f64_i32
#[no_mangle]
pub unsafe fn cast_unchecked_f64_i32(a: f64) -> i32 {
    // CHECK: {{.*}} call {{.*}} @llvm.wasm.trunc.signed.{{.*}}
    // CHECK-NEXT: ret i32 {{.*}}
    a.to_int_unchecked()
}

// CHECK-LABEL: @cast_unchecked_f32_i64
#[no_mangle]
pub unsafe fn cast_unchecked_f32_i64(a: f32) -> i64 {
    // CHECK: {{.*}} call {{.*}} @llvm.wasm.trunc.signed.{{.*}}
    // CHECK-NEXT: ret i64 {{.*}}
    a.to_int_unchecked()
}

// CHECK-LABEL: @cast_unchecked_f32_i32
#[no_mangle]
pub unsafe fn cast_unchecked_f32_i32(a: f32) -> i32 {
    // CHECK: {{.*}} call {{.*}} @llvm.wasm.trunc.signed.{{.*}}
    // CHECK-NEXT: ret i32 {{.*}}
    a.to_int_unchecked()
}

// CHECK-LABEL: @cast_unchecked_f64_u64
#[no_mangle]
pub unsafe fn cast_unchecked_f64_u64(a: f64) -> u64 {
    // CHECK: {{.*}} call {{.*}} @llvm.wasm.trunc.unsigned.{{.*}}
    // CHECK-NEXT: ret i64 {{.*}}
    a.to_int_unchecked()
}

// CHECK-LABEL: @cast_unchecked_f64_u32
#[no_mangle]
pub unsafe fn cast_unchecked_f64_u32(a: f64) -> u32 {
    // CHECK: {{.*}} call {{.*}} @llvm.wasm.trunc.unsigned.{{.*}}
    // CHECK-NEXT: ret i32 {{.*}}
    a.to_int_unchecked()
}

// CHECK-LABEL: @cast_unchecked_f32_u64
#[no_mangle]
pub unsafe fn cast_unchecked_f32_u64(a: f32) -> u64 {
    // CHECK: {{.*}} call {{.*}} @llvm.wasm.trunc.unsigned.{{.*}}
    // CHECK-NEXT: ret i64 {{.*}}
    a.to_int_unchecked()
}

// CHECK-LABEL: @cast_unchecked_f32_u32
#[no_mangle]
pub unsafe fn cast_unchecked_f32_u32(a: f32) -> u32 {
    // CHECK: {{.*}} call {{.*}} @llvm.wasm.trunc.unsigned.{{.*}}
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
