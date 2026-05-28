//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "staticlib"]

// CHECK: define{{.*}}void @a()
#[no_mangle]
#[inline]
pub extern "C" fn a() {}

// CHECK: define{{.*}}void @b()
#[export_name = "b"]
#[inline]
pub extern "C" fn b() {}

// CHECK: define{{.*}}void @c()
#[no_mangle]
#[inline]
extern "C" fn c() {}

// CHECK: define{{.*}}void @d()
#[export_name = "d"]
#[inline]
extern "C" fn d() {}

// CHECK: define{{.*}}void @e()
#[no_mangle]
#[inline(always)]
pub extern "C" fn e() {}

// CHECK: define{{.*}}void @f()
#[export_name = "f"]
#[inline(always)]
pub extern "C" fn f() {}

// CHECK: define{{.*}}void @g()
#[no_mangle]
#[inline(always)]
extern "C" fn g() {}

// CHECK: define{{.*}}void @h()
#[export_name = "h"]
#[inline(always)]
extern "C" fn h() {}
