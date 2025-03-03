//@ add-core-stubs
//@ revisions: aarch64 x86-64
//@ [aarch64] compile-flags: -Ctarget-feature=+neon,+fp16,+fhm --target=aarch64-unknown-linux-gnu
//@ [aarch64] needs-llvm-components: aarch64
//@ [x86-64] compile-flags: -Ctarget-feature=+sse4.2,+rdrand --target=x86_64-unknown-linux-gnu
//@ [x86-64] needs-llvm-components: x86
//@ build-pass
#![no_core]
#![crate_type = "rlib"]
#![feature(intrinsics, rustc_attrs, no_core, lang_items, staged_api)]
#![stable(feature = "test", since = "1.0.0")]

extern crate minicore;
use minicore::*;

#[stable(feature = "test", since = "1.0.0")]
#[rustc_const_stable(feature = "test", since = "1.0.0")]
#[rustc_intrinsic]
const unsafe fn unreachable() -> !;

#[rustc_builtin_macro]
macro_rules! cfg {
    ($($cfg:tt)*) => {};
}

// Test code
const fn do_or_die(cond: bool) {
    if cond {
    } else {
        unsafe { unreachable() }
    }
}

macro_rules! assert {
    ($x:expr $(,)?) => {
        const _: () = do_or_die($x);
    };
}


#[cfg(target_arch = "aarch64")]
fn check_aarch64() {
    // These checks that the rustc feature name is used, not the LLVM feature.

    assert!(cfg!(target_feature = "neon"));
    // #[expect(unexpected_cfgs)] except that 32-bit arm actually use fp-armv8
    { assert!(cfg!(not(target_feature = "fp-armv8"))); }

    assert!(cfg!(target_feature = "fhm"));
    #[expect(unexpected_cfgs)]
    { assert!(cfg!(not(target_feature = "fp16fml"))); }

    assert!(cfg!(target_feature = "fp16"));
    #[expect(unexpected_cfgs)]
    { assert!(cfg!(not(target_feature = "fullfp16"))); }
}

#[cfg(target_arch = "x86_64")]
fn check_x86_64() {
    // This checks that the rustc feature name is used, not the LLVM feature.
    assert!(cfg!(target_feature = "rdrand"));
    #[expect(unexpected_cfgs)]
    { assert!(cfg!(not(target_feature = "rdrnd"))); }

    // Likewise: We enable LLVM's crc32 feature with SSE4.2, but Rust says it's just SSE4.2
    assert!(cfg!(target_feature = "sse4.2"));
    #[expect(unexpected_cfgs)]
    { assert!(cfg!(not(target_feature = "crc32"))); }
}
