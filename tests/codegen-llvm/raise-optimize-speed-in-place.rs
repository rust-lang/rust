//@ revisions: ALL UNIT NOIPA ONE_FILE
//@ compile-flags: -Copt-level=0 -Ccodegen-units=1
//@[ALL] compile-flags: -Zraise-optimize-speed=crate
//@[UNIT] compile-flags: -Zraise-optimize-speed=unit
//@[NOIPA] compile-flags: -Zraise-optimize-speed=crate,noipa
//@[ONE_FILE] compile-flags: -Zraise-optimize-speed=modules

#![crate_type = "rlib"]

// Under the in-place forms of `-Z raise-optimize-speed`, a module of a crate built without
// optimization gets the optimizing pipeline. A function marked `#[optimize(speed)]` is
// optimized there, and every other function is held back with `optnone`.
//
// The forms that give marked functions modules of their own fall back to this where the crate's
// IR must come out as one file, as it must here.

// CHECK-LABEL: define{{.*}}i32 @held_back
// CHECK-SAME: [[HELD_BACK:#[0-9]+]]
// CHECK: alloca
// CHECK: ret i32
#[no_mangle]
pub fn held_back(x: i32) -> i32 {
    let mut y = x;
    y = y.wrapping_add(1);
    y
}

// CHECK-LABEL: define{{.*}}i32 @marked
// CHECK-SAME: [[MARKED:#[0-9]+]]
// CHECK-NOT: alloca
// CHECK: ret i32
#[optimize(speed)]
#[no_mangle]
pub fn marked(x: i32) -> i32 {
    let mut y = x;
    y = y.wrapping_add(1);
    y
}

// ALL: attributes [[HELD_BACK]] = {{.*}}noinline{{.*}}optnone
// UNIT: attributes [[HELD_BACK]] = {{.*}}noinline{{.*}}optnone
// NOIPA: attributes [[HELD_BACK]] = {{.*}}noipa{{.*}}optnone
// ONE_FILE: attributes [[HELD_BACK]] = {{.*}}noinline{{.*}}optnone
// CHECK: attributes [[MARKED]] = {
// CHECK-NOT: optnone
// CHECK-SAME: }
