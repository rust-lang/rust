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
    loop {
        unsafe { maybe_has_side_effect() }
        i += 1;
        if i >= 10 {
            break;
        }
    }
}

// CHECK: ![[HINT]] = distinct !{![[HINT]], ![[INNER_HINT:[0-9]+]]}
// CHECK: ![[INNER_HINT]] = !{!"llvm.loop.unroll.enable"}
