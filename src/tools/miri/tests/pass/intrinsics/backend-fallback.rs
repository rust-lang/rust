#![feature(intrinsics)]

// Having a fallback body that does not match the
// intrinsic is not a good idea, but we do that here for testing.
//
// It's especially a bad idea to mark such a function as
// #[miri::intrinsic_fallback_is_spec].
#[rustc_intrinsic(llvm = "llvm.fshl.i64")]
#[miri::intrinsic_fallback_is_spec]
fn not_quite_fshl(a: i64, b: i64, sh: i64) -> i64 {
    eprintln!("{a} {b} {sh}");
    0
}

fn main() {
    let res = not_quite_fshl(1, 2, 3);
    assert_eq!(res, 0);
}
