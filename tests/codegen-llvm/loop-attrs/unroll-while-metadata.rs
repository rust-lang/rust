//@ compile-flags: -O -Cno-prepopulate-passes

#![crate_type = "lib"]
#![feature(loop_hints)]

// This test ensures that we emit the expected LLVM metadata for loop hint attributes.
// It does not test that loops are optimized as expected, because successful unrolling removes the
// branch and thus the metadata.

unsafe extern "C" {
    fn maybe_has_side_effect();
}

#[no_mangle]
pub fn unroll_hint() {
    // CHECK-LABEL: @unroll_hint
    // CHECK: !llvm.loop ![[HINT:[0-9]+]]
    let mut i = 0;
    #[unroll]
    while i < 10 {
        unsafe { maybe_has_side_effect() }
        i += 1;
    }
}

#[no_mangle]
pub fn unroll_full() {
    // CHECK-LABEL: @unroll_full
    // CHECK: !llvm.loop ![[FULL:[0-9]+]]
    let mut i = 0;
    #[unroll(full)]
    while i < 10 {
        unsafe { maybe_has_side_effect() }
        i += 1;
    }
}

#[no_mangle]
pub fn unroll_never() {
    // CHECK-LABEL: @unroll_never
    // CHECK: !llvm.loop ![[DISABLE:[0-9]+]]
    let mut i = 0;
    #[unroll(never)]
    while i < 10 {
        unsafe { maybe_has_side_effect() }
        i += 1;
    }
}

#[no_mangle]
pub fn unroll_count() {
    // CHECK-LABEL: @unroll_count
    // CHECK: !llvm.loop ![[COUNT:[0-9]+]]
    let mut i = 0;
    #[unroll(5)]
    while i < 10 {
        unsafe { maybe_has_side_effect() }
        i += 1;
    }
}

// CHECK: ![[HINT]] = distinct !{![[HINT]], ![[INNER_HINT:[0-9]+]]}
// CHECK: ![[INNER_HINT]] = !{!"llvm.loop.unroll.enable"}

// CHECK: ![[FULL]] = distinct !{![[FULL]], ![[INNER_FULL:[0-9]+]]}
// CHECK: ![[INNER_FULL]] = !{!"llvm.loop.unroll.full"}

// CHECK: ![[DISABLE]] = distinct !{![[DISABLE]], ![[INNER_DISABLE:[0-9]+]]}
// CHECK: ![[INNER_DISABLE]] = !{!"llvm.loop.unroll.disable"}

// CHECK: ![[COUNT]] = distinct !{![[COUNT]], ![[INNER_COUNT:[0-9]+]]}
// CHECK: ![[INNER_COUNT]] = !{!"llvm.loop.unroll.count", i32 5}
