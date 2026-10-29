//@ run-pass
//@ revisions: default fallback onlyfallback
//@ [onlyfallback]ignore-backends: llvm
//@ [default]needs-backends: llvm
//@ [fallback]needs-backends: llvm
//@ [fallback]compile-flags: -Zforce-intrinsic-fallback

#![feature(intrinsics)]

// Having a fallback body that does not match the
// intrinsic is not a good idea, but we do that here for testing.
#[rustc_intrinsic(llvm = "llvm.fshl.i64")]
const fn sometimes_fshl(a: i64, b: i64, sh: i64) -> i64 {
    a + b + sh
}

#[cfg(default)]
const IS_INTRINSIC: bool = true;

#[cfg(any(fallback, onlyfallback))]
const IS_INTRINSIC: bool = false;

fn main() {
    let res = sometimes_fshl(1, 2, 3);
    if IS_INTRINSIC {
        assert_eq!(res, 8);
    } else {
        assert_eq!(res, 6);
    }

    // Const eval should always use the fallback body, regardless
    // of -Zforce-intrinsic-fallback.
    let const_res = const { sometimes_fshl(1, 2, 3) };
    assert_eq!(const_res, 6);
}
