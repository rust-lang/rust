//@ revisions: aarch64-neon aarch64-sve2
//@ [aarch64-neon] compile-flags: -Ctarget-feature=+neon --target=aarch64-unknown-linux-gnu
//@ [aarch64-neon] needs-llvm-components: aarch64
//@ [aarch64-sve2] compile-flags: -Ctarget-feature=-neon,+sve2 --target=aarch64-unknown-linux-gnu
//@ [aarch64-sve2] needs-llvm-components: aarch64
//@ build-pass
#![no_core]
#![crate_type = "rlib"]
#![feature(intrinsics, rustc_attrs, no_core, lang_items, staged_api)]
#![stable(feature = "test", since = "1.0.0")]

// Tests vetting "feature hierarchies" in the cases where we impose them.

// Supporting minimal rust core code
#[lang = "pointee_sized"]
trait PointeeSized {}

#[lang = "meta_sized"]
trait MetaSized: PointeeSized {}

#[lang = "sized"]
trait Sized: MetaSized {}

#[lang = "copy"]
trait Copy {}

impl Copy for bool {}

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


#[cfg(aarch64_neon)]
fn check_neon_not_sve2() {
    // This checks that a normal aarch64 target doesn't suddenly jump up the feature hierarchy.
    assert!(cfg!(target_feature = "neon"));
    assert!(cfg!(not(target_feature = "sve2")));
}

#[cfg(aarch64_sve2)]
fn check_sve2_includes_neon() {
    // This checks that aarch64's sve2 includes neon
    assert!(cfg!(target_feature = "neon"));
    assert!(cfg!(target_feature = "sve2"));
}
