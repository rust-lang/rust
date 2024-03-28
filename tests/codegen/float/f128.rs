// Verify that our intrinsics generate the correct LLVM calls for f128

#![crate_type = "lib"]
#![feature(f128)]
#![feature(core_intrinsics)]

// CHECK-LABEL: i1 @f128_eq(
#[no_mangle]
pub fn f128_eq(a: f128, b: f128) -> bool {
    // CHECK: fcmp oeq fp128 %{{.+}}, %{{.+}}
    a == b
}

// CHECK-LABEL: i1 @f128_ne(
#[no_mangle]
pub fn f128_ne(a: f128, b: f128) -> bool {
    // CHECK: fcmp une fp128 %{{.+}}, %{{.+}}
    a != b
}

// CHECK-LABEL: i1 @f128_gt(
#[no_mangle]
pub fn f128_gt(a: f128, b: f128) -> bool {
    // CHECK: fcmp ogt fp128 %{{.+}}, %{{.+}}
    a > b
}

// CHECK-LABEL: i1 @f128_ge(
#[no_mangle]
pub fn f128_ge(a: f128, b: f128) -> bool {
    // CHECK: fcmp oge fp128 %{{.+}}, %{{.+}}
    a >= b
}

// CHECK-LABEL: i1 @f128_lt(
#[no_mangle]
pub fn f128_lt(a: f128, b: f128) -> bool {
    // CHECK: fcmp olt fp128 %{{.+}}, %{{.+}}
    a < b
}

// CHECK-LABEL: i1 @f128_le(
#[no_mangle]
pub fn f128_le(a: f128, b: f128) -> bool {
    // CHECK: fcmp ole fp128 %{{.+}}, %{{.+}}
    a <= b
}

// CHECK-LABEL: fp128 @f128_neg(
#[no_mangle]
pub fn f128_neg(a: f128) -> f128 {
    // CHECK: fneg fp128
    -a
}

// CHECK-LABEL: fp128 @f128_add(
#[no_mangle]
pub fn f128_add(a: f128, b: f128) -> f128 {
    // CHECK: fadd fp128 %{{.+}}, %{{.+}}
    a + b
}

// CHECK-LABEL: fp128 @f128_sub(
#[no_mangle]
pub fn f128_sub(a: f128, b: f128) -> f128 {
    // CHECK: fsub fp128 %{{.+}}, %{{.+}}
    a - b
}

// CHECK-LABEL: fp128 @f128_mul(
#[no_mangle]
pub fn f128_mul(a: f128, b: f128) -> f128 {
    // CHECK: fmul fp128 %{{.+}}, %{{.+}}
    a * b
}

// CHECK-LABEL: fp128 @f128_div(
#[no_mangle]
pub fn f128_div(a: f128, b: f128) -> f128 {
    // CHECK: fdiv fp128 %{{.+}}, %{{.+}}
    a / b
}

// CHECK-LABEL: fp128 @f128_rem(
#[no_mangle]
pub fn f128_rem(a: f128, b: f128) -> f128 {
    // CHECK: frem fp128 %{{.+}}, %{{.+}}
    a % b
}

// CHECK-LABEL: void @f128_add_assign(
#[no_mangle]
pub fn f128_add_assign(a: &mut f128, b: f128) {
    // CHECK: fadd fp128 %{{.+}}, %{{.+}}
    // CHECK-NEXT: store fp128 %{{.+}}, ptr %{{.+}}
    *a += b;
}

// CHECK-LABEL: void @f128_sub_assign(
#[no_mangle]
pub fn f128_sub_assign(a: &mut f128, b: f128) {
    // CHECK: fsub fp128 %{{.+}}, %{{.+}}
    // CHECK-NEXT: store fp128 %{{.+}}, ptr %{{.+}}
    *a -= b;
}

// CHECK-LABEL: void @f128_mul_assign(
#[no_mangle]
pub fn f128_mul_assign(a: &mut f128, b: f128) {
    // CHECK: fmul fp128 %{{.+}}, %{{.+}}
    // CHECK-NEXT: store fp128 %{{.+}}, ptr %{{.+}}
    *a *= b
}

// CHECK-LABEL: void @f128_div_assign(
#[no_mangle]
pub fn f128_div_assign(a: &mut f128, b: f128) {
    // CHECK: fdiv fp128 %{{.+}}, %{{.+}}
    // CHECK-NEXT: store fp128 %{{.+}}, ptr %{{.+}}
    *a /= b
}

// CHECK-LABEL: void @f128_rem_assign(
#[no_mangle]
pub fn f128_rem_assign(a: &mut f128, b: f128) {
    // CHECK: frem fp128 %{{.+}}, %{{.+}}
    // CHECK-NEXT: store fp128 %{{.+}}, ptr %{{.+}}
    *a %= b
}
