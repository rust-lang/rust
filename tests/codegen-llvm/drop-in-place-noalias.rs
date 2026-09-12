//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes
//
// LLVM23 changed the meanign of "dereferenceable", allowing us to apply it in more cases,
// see <https://github.com/rust-lang/rust/pull/158863>.
//@ revisions: LLVM22 LLVM23
//@ [LLVM22] max-llvm-major-version: 22
//@ [LLVM23] min-llvm-version: 23

// Tests that the compiler can apply `noalias` and other &mut attributes to `drop_in_place`.
// Note that non-Unpin types should not get `noalias`, matching &mut behavior.

#![crate_type = "lib"]

use std::marker::PhantomPinned;

// CHECK: define internal void @{{.*}}core{{.*}}ptr{{.*}}drop_in_place{{.*}}StructUnpin{{.*}}(ptr noalias nofree noundef align 4 dereferenceable(12) %{{.+}})

// LLVM22: define internal void @{{.*}}core{{.*}}ptr{{.*}}drop_in_place{{.*}}StructNotUnpin{{.*}}(ptr noundef nonnull align 4 %{{.+}})
// LLVM23: define internal void @{{.*}}core{{.*}}ptr{{.*}}drop_in_place{{.*}}StructNotUnpin{{.*}}(ptr noundef align 4 dereferenceable(12) %{{.+}})

pub struct StructUnpin {
    a: i32,
    b: i32,
    c: i32,
}

impl Drop for StructUnpin {
    fn drop(&mut self) {}
}

pub struct StructNotUnpin {
    a: i32,
    b: i32,
    c: i32,
    p: PhantomPinned,
}

impl Drop for StructNotUnpin {
    fn drop(&mut self) {}
}

pub unsafe fn main(x: StructUnpin, y: StructNotUnpin) {
    drop(x);
    drop(y);
}
