//@ run-pass
//@ revisions: backend_with_support backend_without_support force_fallback
//@ [backend_with_support]needs-backends: llvm
//@ [backend_without_support]ignore-backends: llvm
//@ [force_fallback]compile-flags: -Zforce-intrinsic-fallback

#![feature(intrinsics)]

// Having a fallback body that does not match the
// intrinsic is not a good idea, but we do that here for testing.
#[rustc_intrinsic(llvm = "llvm.fshl.i64")]
const fn sometimes_fshl(a: i64, b: i64, sh: i64) -> i64 {
    a + b + sh
}

#[cfg(backend_with_support)]
#[rustc_intrinsic(llvm = "llvm.fshl.i64")]
fn fshl_no_fallback(a: i64, b: i64, sh: i64) -> i64;

// Test a built-in intrinsic name to make sure no
// implementation gets confused.
#[rustc_intrinsic(llvm = "llvm.fshl.i64")]
const fn size_of_val(a: i64, b: i64, sh: i64) -> i64 {
    (((((a as i128) << 64) | (b as i128)) << sh) >> 64) as i64
}

#[cfg(backend_with_support)]
const IS_INTRINSIC: bool = true;

#[cfg(any(backend_without_support, force_fallback))]
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

    #[cfg(backend_with_support)]
    {
        let res = fshl_no_fallback(1, 2, 3);
        assert_eq!(res, 8);
    }

    assert_eq!(const { size_of_val(1, 2, 3) }, 8);
    assert_eq!(size_of_val(1, 2, 3), 8);
}
