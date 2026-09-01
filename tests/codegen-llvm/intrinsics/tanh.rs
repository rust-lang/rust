//@ revisions: DEV OPT
//@ [DEV] compile-flags: -C no-prepopulate-passes
//@ [OPT] compile-flags: -O

#![crate_type = "lib"]

// CHECK-LABEL: @tanhf64
#[no_mangle]
pub fn tanhf64(x: f64) -> f64 {
    // CHECK-NOT: call double @llvm.tanh.f64
    // CHECK: call double @tanh
    x.tanh()
}

// CHECK-LABEL: @tanhf32
#[no_mangle]
pub fn tanhf32(x: f32) -> f32 {
    // CHECK-NOT: call float @llvm.tanh.f32
    // CHECK: call float @tanh
    x.tanh()
}

// CHECK-LABEL: @dce_tanh
#[no_mangle]
pub fn dce_tanh(x: f64) -> f64 {
    // DEV: call double @tanh
    // OPT: call double @tanh
    let _ = x.tanh();
    1.0
}

// CHECK-LABEL: @speculate
#[no_mangle]
pub fn speculate(x: f64, cond: bool) -> f64 {
    // DEV: br i1 %cond
    // OPT: br i1 %cond
    if cond { x.tanh() } else { 0.0 }
}


