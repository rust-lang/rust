//@ revisions: DEV OPT
//@ [DEV] compile-flags: -C no-prepopulate-passes
//@ [OPT] compile-flags: -O

// We previously lowered tanh to libm calls, which would set errno and thus prevent a couple of
// optimizations, since LLVM would mark the call as `memory(errnomem: write)`.
// LLVM since gained additional math intrinsics, including tanh, to which we're now lowering
// instead. They are marked as `readnone`, and thus allow the optimizations shown below.

#![crate_type = "lib"]

// CHECK-LABEL: @tanhf64
#[no_mangle]
pub fn tanhf64(x: f64) -> f64 {
    // CHECK: call double @llvm.tanh.f64
    // CHECK-NOT: call double @tanh(
    x.tanh()
}

// CHECK-LABEL: @tanhf32
#[no_mangle]
pub fn tanhf32(x: f32) -> f32 {
    // CHECK: call float @llvm.tanh.f32
    // CHECK-NOT: call float @tanhf(
    x.tanh()
}

// Since we're now marked as readnone, the call to tanh can be optimized away
// CHECK-LABEL: @dce_tanh
#[no_mangle]
pub fn dce_tanh(x: f64) -> f64 {
    // DEV: call double @llvm.tanh.f64
    // OPT-NOT: llvm.tanh
    let _ = x.tanh();
    1.0
}

// Since we're now marked as speculate, we can fold the branch into a select
// CHECK-LABEL: @speculate
#[no_mangle]
pub fn speculate(x: f64, cond: bool) -> f64 {
    // DEV: br i1 %cond
    // OPT-NOT: br i1
    // OPT: %0 = tail call double @llvm.tanh.f64(double %x)
    // OPT-NEXT: %_0.sroa.0.0 = select i1 %cond, double %0, double 0.000000e+00
    // OPT-NEXT: ret double %_0.sroa.0.0
    if cond { x.tanh() } else { 0.0 }
}
