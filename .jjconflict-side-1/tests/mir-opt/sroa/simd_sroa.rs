//@ needs-unwind
#![feature(portable_simd)]

// SRoA expands things even if they're unused
// <https://github.com/rust-lang/rust/issues/144621>

use std::simd::Simd;

// EMIT_MIR simd_sroa.foo.ScalarReplacementOfAggregates.diff
pub(crate) fn foo(simds: &[Simd<u8, 16>], _unused: Simd<u8, 16>) {
    // CHECK-LABEL: fn foo
    // CHECK-NOT: [u8; 16]
    // CHECK: let [[SIMD:_.+]]: std::simd::Simd<u8, 16>;
    // CHECK-NOT: [u8; 16]
    // CHECK: [[SIMD]] = copy (*_1)[0 of 1];
    // CHECK-NOT: [u8; 16]
    let a = simds[0];
}
