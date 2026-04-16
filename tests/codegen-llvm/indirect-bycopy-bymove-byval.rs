//! Regression test for issue <https://github.com/rust-lang/rust/issues/155241>.
//! Arguments passed indirectly via a hidden pointer must be copied to an alloca,
//! except for by-val or by-move.
//@ add-minicore
//@ revisions: x64-linux i686-linux i686-windows
//@ compile-flags: -Cno-prepopulate-passes -Copt-level=3
//@[x64-linux] compile-flags: --target x86_64-unknown-linux-gnu
//@[x64-linux] needs-llvm-components: x86
//@[i686-linux] compile-flags: --target i686-unknown-linux-gnu
//@[i686-linux] needs-llvm-components: x86
//@[i686-windows] compile-flags: --target i686-pc-windows-msvc
//@[i686-windows] needs-llvm-components: x86

#![crate_type = "lib"]
#![feature(stmt_expr_attributes, no_core)]
#![expect(unused)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

struct Thing(u64, u64, u64);

impl Copy for Thing {}

// The argument of the second call is a by-move argument.

// CHECK-LABEL: @normal
// CHECK: call void @llvm.memcpy{{.*}}(ptr{{.*}} [[normal_V1:%.*]], ptr{{.*}} %value,
// CHECK: call void @opaque(ptr{{.*}} [[normal_V1]])
// CHECK: call void @opaque(ptr{{.*}} %value)
// CHECK: call void @llvm.memcpy{{.*}}(ptr{{.*}} [[normal_V3:%.*]], ptr{{.*}} @anon{{.*}},
// CHECK: call void @opaque(ptr{{.*}} [[normal_V3]])
#[unsafe(no_mangle)]
pub fn normal() {
    #[inline(never)]
    #[unsafe(no_mangle)]
    fn opaque(mut thing: Thing) {
        thing.0 = 1;
    }
    let value = Thing(0, 0, 0);
    opaque(value);
    opaque(value);
    const VALUE: Thing = Thing(0, 0, 0);
    opaque(VALUE);
}

// FIXME: closure#0 and closure#1 are missing memcpy.

// CHECK-LABEL: @untupled
// CHECK: call indirect_bycopy_bymove_byval::untupled::{closure#0}
// CHECK-NEXT: call void @{{.*}}(ptr {{.*}}, ptr{{.*}} %value)
// CHECK: call indirect_bycopy_bymove_byval::untupled::{closure#1}
// CHECK-NEXT: call void @{{.*}}(ptr {{.*}}, ptr{{.*}} %value)
// CHECK: call indirect_bycopy_bymove_byval::untupled::{closure#2}
// CHECK-NEXT: call void @{{.*}}(ptr {{.*}}, ptr{{.*}} @anon{{.*}})
#[unsafe(no_mangle)]
pub fn untupled() {
    let value = (Thing(0, 0, 0),);
    (#[inline(never)]
    |mut thing: Thing| {
        thing.0 = 1;
    })
    .call(value);
    (#[inline(never)]
    |mut thing: Thing| {
        thing.0 = 2;
    })
    .call(value);
    const VALUE: (Thing,) = (Thing(0, 0, 0),);
    (#[inline(never)]
    |mut thing: Thing| {
        thing.0 = 3;
    })
    .call(VALUE);
}

// FIXME: all memcpy calls are redundant for byval.

// CHECK-LABEL: @byval
// CHECK: call void @llvm.memcpy{{.*}}(ptr{{.*}} [[byval_V1:%.*]], ptr{{.*}} %value,
// CHECK: call void @opaque_byval(ptr{{.*}} byval([24 x i8]){{.*}} [[byval_V1]])
// CHECK: call void @opaque_byval(ptr{{.*}} byval([24 x i8]){{.*}} %value)
// CHECK: call void @llvm.memcpy{{.*}}(ptr{{.*}} [[byval_V3:%.*]], ptr{{.*}} @anon{{.*}},
// CHECK: call void @opaque_byval(ptr{{.*}} byval([24 x i8]){{.*}} [[byval_V3]])
#[unsafe(no_mangle)]
pub fn byval() {
    #[repr(C)]
    struct Thing(u64, u64, u64);
    impl Copy for Thing {}
    #[inline(never)]
    #[unsafe(no_mangle)]
    extern "C" fn opaque_byval(mut thing: Thing) {
        thing.0 = 1;
    }
    let value = Thing(0, 0, 0);
    opaque_byval(value);
    opaque_byval(value);
    const VALUE: Thing = Thing(0, 0, 0);
    opaque_byval(VALUE);
}
