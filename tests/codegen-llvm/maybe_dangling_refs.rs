// This test checks for absence of noalias and dereferenceable attributes on
// arguments wrapped in `MaybeDangling`.
//
// This also tests
//
//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled -Cno-prepopulate-passes
#![crate_type = "lib"]
#![feature(maybe_dangling)]

use std::mem::MaybeDangling;

// CHECK: define {{(dso_local )?}}noundef nonnull ptr @f(ptr noundef nonnull %x) unnamed_addr
#[no_mangle]
pub fn f(x: MaybeDangling<Box<u8>>) -> MaybeDangling<Box<u8>> {
    x
}

// CHECK: define {{(dso_local )?}}noundef nonnull ptr @g(ptr noundef nonnull %x) unnamed_addr
#[no_mangle]
pub fn g(x: MaybeDangling<&u8>) -> MaybeDangling<&u8> {
    x
}

// CHECK: define {{(dso_local )?}}noundef nonnull ptr @h(ptr noundef nonnull %x) unnamed_addr
#[no_mangle]
pub fn h(x: MaybeDangling<&mut u8>) -> MaybeDangling<&mut u8> {
    x
}

// CHECK: define {{(dso_local )?}}noundef nonnull align 4 ptr @i(ptr noundef nonnull align 4 %x) unnamed_addr
#[no_mangle]
pub fn i(x: MaybeDangling<Box<u32>>) -> MaybeDangling<Box<u32>> {
    x
}

// CHECK: define {{(dso_local )?}}noundef nonnull align 4 ptr @j(ptr noundef nonnull align 4 %x) unnamed_addr
#[no_mangle]
pub fn j(x: MaybeDangling<&u32>) -> MaybeDangling<&u32> {
    x
}

// CHECK: define {{(dso_local )?}}noundef nonnull align 4 ptr @k(ptr noundef nonnull align 4 %x) unnamed_addr
#[no_mangle]
pub fn k(x: MaybeDangling<&mut u32>) -> MaybeDangling<&mut u32> {
    x
}
