//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "cdylib"]

// CHECK: define{{( dso_local)?}} void @a()
#[no_mangle]
#[inline]
pub extern "C" fn a() {}

// CHECK: define{{( dso_local)?}} void @b()
#[export_name = "b"]
#[inline]
pub extern "C" fn b() {}

// CHECK: define{{( dso_local)?}} void @c()
#[no_mangle]
#[inline]
extern "C" fn c() {}

// CHECK: define{{( dso_local)?}} void @d()
#[export_name = "d"]
#[inline]
extern "C" fn d() {}

// CHECK: define{{( dso_local)?}} void @e()
#[no_mangle]
#[inline(always)]
pub extern "C" fn e() {}

// CHECK: define{{( dso_local)?}} void @f()
#[export_name = "f"]
#[inline(always)]
pub extern "C" fn f() {}

// CHECK: define{{( dso_local)?}} void @g()
#[no_mangle]
#[inline(always)]
extern "C" fn g() {}

// CHECK: define{{( dso_local)?}} void @h()
#[export_name = "h"]
#[inline(always)]
extern "C" fn h() {}
