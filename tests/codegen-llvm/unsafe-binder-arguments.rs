//@ compile-flags: -Znext-solver -Copt-level=3 -C no-prepopulate-passes

// Validity invariants for unsafe binders are currently not fully specified.
//
// Today a bare binder around a reference only gets `nonnull`, while references
// in an aggregate under a binder get the same attributes as without the binder.

#![crate_type = "lib"]
#![feature(unsafe_binders)]
#![allow(incomplete_features)]

// CHECK: define noundef nonnull ptr @bare(ptr noundef nonnull %x)
#[no_mangle]
pub fn bare(x: unsafe<'a> &'a u32) -> unsafe<'a> &'a u32 {
    x
}

// CHECK: define void @bare_mut(ptr noundef nonnull %_x)
#[no_mangle]
pub fn bare_mut(_x: unsafe<'a> &'a mut u32) {}

// CHECK: define void @pair(
// CHECK-SAME: ptr noalias nofree noundef readonly align 4 {{.*}}dereferenceable(4) %_x.0,
// CHECK-SAME: ptr noalias nofree noundef readonly align 8 {{.*}}dereferenceable(8) %_x.1)
#[no_mangle]
pub fn pair(_x: unsafe<'a> (&'a u32, &'a u64)) {}
