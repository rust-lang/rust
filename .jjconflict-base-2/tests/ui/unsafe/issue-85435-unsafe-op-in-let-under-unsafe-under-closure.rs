//@ check-pass

// This is issue #85435. But the real story is reflected in issue #85561, where
// a bug in the implementation of feature(capture_disjoint_fields) () was
// exposed to non-feature-gated code by a diagnostic changing PR that removed
// the gating in one case.

// This test is double-checking that the case of interest continues to work as
// expected in the *absence* of that feature gate. At the time of this writing,
// enabling the feature gate will cause this test to fail. We obviously cannot
// stabilize that feature until it can correctly handle this test.

fn main() {
    let val: u8 = 5;
    let u8_ptr: *const u8 = &val;
    let _closure = || {
        unsafe {
            let tmp = *u8_ptr;
            tmp

            // Just dereferencing and returning directly compiles fine:
            // *u8_ptr
        }
    };
}
