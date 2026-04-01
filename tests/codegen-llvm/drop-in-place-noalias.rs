//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes

// Tests that the compiler can apply `noalias` and other &mut attributes to `drop_in_place`.
// Note that non-Unpin types should not get `noalias`, matching &mut behavior.

#![crate_type = "lib"]

use std::marker::PhantomPinned;

// CHECK: define internal void @{{.*}}core{{.*}}ptr{{.*}}drop_in_place{{.*}}StructUnpin{{.*}}(ptr noalias noundef align 4 dereferenceable(12) %{{.+}})

// CHECK: define internal void @{{.*}}core{{.*}}ptr{{.*}}drop_in_place{{.*}}StructNotUnpin{{.*}}(ptr noundef nonnull align 4 %{{.+}})

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
