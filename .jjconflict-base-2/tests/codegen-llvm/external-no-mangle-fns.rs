//@ compile-flags: -C no-prepopulate-passes
// `#[no_mangle]`d functions always have external linkage, i.e., no `internal` in their `define`s

#![crate_type = "lib"]
#![no_std]

// CHECK: define{{( dso_local)?}} void @a()
#[no_mangle]
fn a() {}

// CHECK: define{{( dso_local)?}} void @b()
#[no_mangle]
pub fn b() {}

mod private {
    // CHECK: define{{( dso_local)?}} void @c()
    #[no_mangle]
    fn c() {}

    // CHECK: define{{( dso_local)?}} void @d()
    #[no_mangle]
    pub fn d() {}
}

const HIDDEN: () = {
    // CHECK: define{{( dso_local)?}} void @e()
    #[no_mangle]
    fn e() {}

    // CHECK: define{{( dso_local)?}} void @f()
    #[no_mangle]
    pub fn f() {}
};

// The surrounding item should not accidentally become external
// CHECK-LABEL: ; external_no_mangle_fns::x
// CHECK-NEXT: ; Function Attrs:
// CHECK-NEXT: define internal
#[inline(never)]
fn x() {
    // CHECK: define{{( dso_local)?}} void @g()
    #[no_mangle]
    fn g() {
        x();
    }

    // CHECK: define{{( dso_local)?}} void @h()
    #[no_mangle]
    pub fn h() {}

    // side effect to keep `x` around
    unsafe {
        core::ptr::read_volatile(&42);
    }
}

// CHECK: define{{( dso_local)?}} void @i()
#[no_mangle]
#[inline]
fn i() {}

// CHECK: define{{( dso_local)?}} void @j()
#[no_mangle]
#[inline]
pub fn j() {}

// CHECK: define{{( dso_local)?}} void @k()
#[no_mangle]
#[inline(always)]
fn k() {}

// CHECK: define{{( dso_local)?}} void @l()
#[no_mangle]
#[inline(always)]
pub fn l() {}
