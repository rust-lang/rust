// compile-flags: -C no-prepopulate-passes

#![crate_type = "cdylib"]

// CHECK: define void @a()
#[no_mangle]
#[inline]
pub extern "C" fn a() {}

// CHECK: define void @b()
#[export_name = "b"]
#[inline]
pub extern "C" fn b() {}

// CHECK: define void @c()
#[no_mangle]
#[inline]
extern "C" fn c() {}

// CHECK: define void @d()
#[export_name = "d"]
#[inline]
extern "C" fn d() {}
