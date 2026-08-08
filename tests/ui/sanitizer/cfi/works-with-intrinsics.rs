// Verifies that intrinsics and LLVM intrinsics can be called.
//
//@ needs-sanitizer-cfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Ccodegen-units=1 -Clto -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=cfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

#![feature(abi_unadjusted, link_llvm_intrinsics)]
#![allow(internal_features)]

unsafe extern "unadjusted" {
    #[link_name = "llvm.bitreverse.i32"]
    fn bitreverse(x: i32) -> i32;
}

fn main() {
    // Intrinsics
    // The black_box intrinsic (i.e., a fn item with #[rustc_intrinsic]) is not transformed, as
    // it can not be reified or called indirectly.
    assert_eq!(std::hint::black_box(1i32), 1);

    // LLVM intrinsics
    // The bitreverse LLVM intrinsic (i.e., a fn item with extern "unadjusted") is not
    // transformed, as it can not be reified or called indirectly.
    assert_eq!(unsafe { bitreverse(1i32) }, i32::MIN);
}
