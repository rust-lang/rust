//@ only-aarch64
//@ needs-unwind
#![feature(stdarch_aarch64_sve)]

// SRoA expands things even if they're unused
// <https://github.com/rust-lang/rust/issues/144621>

use std::arch::aarch64::{svuint32_t, svuint32x2_t};

// EMIT_MIR scalable_sroa.foo.ScalarReplacementOfAggregates.diff
pub(crate) fn foo(simds: &[svuint32_t], _unused: svuint32_t) {
    // CHECK-LABEL: fn foo
    // CHECK-NOT: u32
    // CHECK: let [[SIMD:_.+]]: std::arch::aarch64::svuint32_t;
    // CHECK-NOT: u32
    // CHECK: [[SIMD]] = copy (*_1)[0 of 1];
    // CHECK-NOT: u32
    let a = simds[0];
}

// EMIT_MIR scalable_sroa.bar.ScalarReplacementOfAggregates.diff
pub(crate) fn bar(simds: &[svuint32x2_t], _unused: svuint32x2_t) {
    // CHECK-LABEL: fn bar
    // CHECK-NOT: { <vscale x u32 x 4>, <vscale x u32 x 4> }
    // CHECK: let [[SIMD:_.+]]: std::arch::aarch64::svuint32x2_t;
    // CHECK-NOT: { <vscale x u32 x 4>, <vscale x u32 x 4> }
    // CHECK: [[SIMD]] = copy (*_1)[0 of 1];
    // CHECK-NOT: { <vscale x u32 x 4>, <vscale x u32 x 4> }
    let a = simds[0];
}
