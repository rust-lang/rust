// Verify that our intrinsics generate the correct LLVM calls for f16

#![crate_type = "lib"]
#![feature(f16)]
#![feature(core_intrinsics)]

// CHECK-LABEL: i1 @f16_eq(
#[no_mangle]
pub fn f16_eq(a: f16, b: f16) -> bool {
    // CHECK: fcmp oeq half %{{.+}}, %{{.+}}
    a == b
}

// CHECK-LABEL: i1 @f16_ne(
#[no_mangle]
pub fn f16_ne(a: f16, b: f16) -> bool {
    // CHECK: fcmp une half %{{.+}}, %{{.+}}
    a != b
}

// CHECK-LABEL: i1 @f16_gt(
#[no_mangle]
pub fn f16_gt(a: f16, b: f16) -> bool {
    // CHECK: fcmp ogt half %{{.+}}, %{{.+}}
    a > b
}

// CHECK-LABEL: i1 @f16_ge(
#[no_mangle]
pub fn f16_ge(a: f16, b: f16) -> bool {
    // CHECK: fcmp oge half %{{.+}}, %{{.+}}
    a >= b
}

// CHECK-LABEL: i1 @f16_lt(
#[no_mangle]
pub fn f16_lt(a: f16, b: f16) -> bool {
    // CHECK: fcmp olt half %{{.+}}, %{{.+}}
    a < b
}

// CHECK-LABEL: i1 @f16_le(
#[no_mangle]
pub fn f16_le(a: f16, b: f16) -> bool {
    // CHECK: fcmp ole half %{{.+}}, %{{.+}}
    a <= b
}

// CHECK-LABEL: half @f16_neg(
#[no_mangle]
pub fn f16_neg(a: f16) -> f16 {
    // CHECK: fneg half %{{.+}}
    -a
}

// CHECK-LABEL: half @f16_add(
#[no_mangle]
pub fn f16_add(a: f16, b: f16) -> f16 {
    // CHECK: fadd half %{{.+}}, %{{.+}}
    a + b
}

// CHECK-LABEL: half @f16_sub(
#[no_mangle]
pub fn f16_sub(a: f16, b: f16) -> f16 {
    // CHECK: fsub half %{{.+}}, %{{.+}}
    a - b
}

// CHECK-LABEL: half @f16_mul(
#[no_mangle]
pub fn f16_mul(a: f16, b: f16) -> f16 {
    // CHECK: fmul half %{{.+}}, %{{.+}}
    a * b
}

// CHECK-LABEL: half @f16_div(
#[no_mangle]
pub fn f16_div(a: f16, b: f16) -> f16 {
    // CHECK: fdiv half %{{.+}}, %{{.+}}
    a / b
}

// CHECK-LABEL: half @f16_rem(
#[no_mangle]
pub fn f16_rem(a: f16, b: f16) -> f16 {
    // CHECK: frem half %{{.+}}, %{{.+}}
    a % b
}

// CHECK-LABEL: void @f16_add_assign(
#[no_mangle]
pub fn f16_add_assign(a: &mut f16, b: f16) {
    // CHECK: fadd half %{{.+}}, %{{.+}}
    // CHECK-NEXT: store half %{{.+}}, ptr %{{.+}}
    *a += b;
}

// CHECK-LABEL: void @f16_sub_assign(
#[no_mangle]
pub fn f16_sub_assign(a: &mut f16, b: f16) {
    // CHECK: fsub half %{{.+}}, %{{.+}}
    // CHECK-NEXT: store half %{{.+}}, ptr %{{.+}}
    *a -= b;
}

// CHECK-LABEL: void @f16_mul_assign(
#[no_mangle]
pub fn f16_mul_assign(a: &mut f16, b: f16) {
    // CHECK: fmul half %{{.+}}, %{{.+}}
    // CHECK-NEXT: store half %{{.+}}, ptr %{{.+}}
    *a *= b
}

// CHECK-LABEL: void @f16_div_assign(
#[no_mangle]
pub fn f16_div_assign(a: &mut f16, b: f16) {
    // CHECK: fdiv half %{{.+}}, %{{.+}}
    // CHECK-NEXT: store half %{{.+}}, ptr %{{.+}}
    *a /= b
}

// CHECK-LABEL: void @f16_rem_assign(
#[no_mangle]
pub fn f16_rem_assign(a: &mut f16, b: f16) {
    // CHECK: frem half %{{.+}}, %{{.+}}
    // CHECK-NEXT: store half %{{.+}}, ptr %{{.+}}
    *a %= b
}
