// Verifies that intrinsics and LLVM intrinsics can be called.
//
//@ needs-sanitizer-kcfi
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Cpanic=abort -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=kcfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

#![feature(link_llvm_intrinsics)]
#![allow(internal_features)]

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.bitreverse.i32"]
    fn bitreverse(x: i32) -> i32;
}

fn main() {
    // Intrinsics
    // The black_box intrinsic (i.e., a fn item with #[rustc_intrinsic]) is lowered by codegen
    // and does not have its own callable MIR, and its fallback body uses InstanceKind::Item, so
    // InstanceKind::Intrinsic is not transformed.
    assert_eq!(std::hint::black_box(1i32), 1);

    // LLVM intrinsics
    // The bitreverse LLVM intrinsic (i.e., a fn item with extern "llvm-intrinsic") is lowered
    // by codegen and does not have its own callable MIR, so it is not transformed.
    assert_eq!(unsafe { bitreverse(1i32) }, i32::MIN);
}
