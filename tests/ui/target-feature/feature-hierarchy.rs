//@ revisions: aarch64-neon aarch64-sve2
//@ revisions: hexagon-v60 hexagon-v68 hexagon-hvxv66
//@ [aarch64-neon] compile-flags: -Ctarget-feature=+neon --target=aarch64-unknown-linux-gnu
//@ [aarch64-neon] needs-llvm-components: aarch64
//@ [aarch64-sve2] compile-flags: -Ctarget-feature=-neon,+sve2 --target=aarch64-unknown-linux-gnu
//@ [aarch64-sve2] needs-llvm-components: aarch64
//@ [hexagon-v60] compile-flags: -Ctarget-feature=+v60 --target=hexagon-unknown-linux-musl
//@ [hexagon-v60] needs-llvm-components: hexagon
//@ [hexagon-v68] compile-flags: -Ctarget-feature=+v68 --target=hexagon-unknown-linux-musl
//@ [hexagon-v68] needs-llvm-components: hexagon
//@ [hexagon-hvxv66] compile-flags: -Ctarget-feature=+hvxv66 --target=hexagon-unknown-linux-musl
//@ [hexagon-hvxv66] needs-llvm-components: hexagon
//@ build-pass
//@ add-minicore
//@ ignore-backends: gcc
#![no_core]
#![crate_type = "rlib"]
#![feature(intrinsics, rustc_attrs, no_core, staged_api)]
#![cfg_attr(any(hexagon_v60, hexagon_v68, hexagon_hvxv66), feature(hexagon_target_feature))]
#![stable(feature = "test", since = "1.0.0")]

// Tests vetting "feature hierarchies" in the cases where we impose them.

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

//[hexagon-v60]~? WARN unstable feature specified for `-Ctarget-feature`: `v60`
//[hexagon-v68]~? WARN unstable feature specified for `-Ctarget-feature`: `v68`
//[hexagon-hvxv66]~? WARN unstable feature specified for `-Ctarget-feature`: `hvxv66`

#[cfg(hexagon_v60)]
fn check_v60_not_v68() {
    // Enabling v60 should not jump up the scalar feature hierarchy.
    assert!(cfg!(target_feature = "v60"));
    assert!(cfg!(not(target_feature = "v62")));
    assert!(cfg!(not(target_feature = "v68")));
}

#[cfg(hexagon_v68)]
fn check_v68_implies_v60() {
    // v68 implies all lower scalar arch versions.
    assert!(cfg!(target_feature = "v60"));
    assert!(cfg!(target_feature = "v62"));
    assert!(cfg!(target_feature = "v65"));
    assert!(cfg!(target_feature = "v66"));
    assert!(cfg!(target_feature = "v67"));
    assert!(cfg!(target_feature = "v68"));
    assert!(cfg!(not(target_feature = "v69")));
}

#[cfg(hexagon_hvxv66)]
fn check_hvxv66_implies_hvx_and_zreg() {
    // hvxv66 implies hvx, hvxv60..v65, and zreg.
    assert!(cfg!(target_feature = "hvx"));
    assert!(cfg!(target_feature = "hvxv60"));
    assert!(cfg!(target_feature = "hvxv62"));
    assert!(cfg!(target_feature = "hvxv65"));
    assert!(cfg!(target_feature = "hvxv66"));
    assert!(cfg!(target_feature = "zreg"));
    assert!(cfg!(not(target_feature = "hvxv67")));
}
