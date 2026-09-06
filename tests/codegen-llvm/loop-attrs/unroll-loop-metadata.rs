//@ compile-flags: -O -Cno-prepopulate-passes

#![crate_type = "lib"]
#![feature(loop_hints)]
#![feature(stmt_expr_attributes)]

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
    #[rustc_unroll]
    loop {
        unsafe { maybe_has_side_effect() }
        i += 1;
        if i >= 10 {
            break;
        }
    }
}

// HIR for a `loop` statement is a bit different, make sure we still apply the metadata in that
// case.

#[no_mangle]
pub fn unroll_full() {
    // CHECK-LABEL: @unroll_full
    // CHECK: !llvm.loop ![[FULL:[0-9]+]]
    let mut i = 0;
    let _return = (#[rustc_unroll(full)]
    loop {
        unsafe { maybe_has_side_effect() }
        i += 1;
        if i >= 10 {
            break 1;
        }
    });
}

#[no_mangle]
pub fn unroll_never() {
    // CHECK-LABEL: @unroll_never
    // CHECK: !llvm.loop ![[DISABLE:[0-9]+]]
    let mut i = 0;
    let _return = (1 + #[rustc_unroll(never)]
    loop {
        unsafe { maybe_has_side_effect() }
        i += 1;
        if i >= 10 {
            break 1;
        }
    });
}

#[no_mangle]
pub fn unroll_count() {
    // CHECK-LABEL: @unroll_count
    // CHECK: !llvm.loop ![[COUNT:[0-9]+]]
    let mut i = 0;
    #[rustc_unroll(5)]
    loop {
        unsafe { maybe_has_side_effect() }
        i += 1;
        if i >= 10 {
            break;
        }
    }
    unsafe { maybe_has_side_effect() }
}

// CHECK-DAG: ![[HINT]] = distinct !{![[HINT]], ![[INNER_HINT:[0-9]+]]}
// CHECK-DAG: ![[INNER_HINT]] = !{!"llvm.loop.unroll.enable"}

// CHECK-DAG: ![[FULL]] = distinct !{![[FULL]], ![[INNER_FULL:[0-9]+]]}
// CHECK-DAG: ![[INNER_FULL]] = !{!"llvm.loop.unroll.full"}

// CHECK-DAG: ![[DISABLE]] = distinct !{![[DISABLE]], ![[INNER_DISABLE:[0-9]+]]}
// CHECK-DAG: ![[INNER_DISABLE]] = !{!"llvm.loop.unroll.disable"}

// CHECK-DAG: ![[COUNT]] = distinct !{![[COUNT]], ![[INNER_COUNT:[0-9]+]]}
// CHECK-DAG: ![[INNER_COUNT]] = !{!"llvm.loop.unroll.count", i32 5}
