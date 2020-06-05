// compile-flags: -C no-prepopulate-passes

#![crate_type = "staticlib"]

// CHECK: define void @a()
#[no_mangle]
#[inline]
pub extern "C" fn a() {
    // side effect to keep `a` around
    unsafe {
        core::ptr::read_volatile(&42);
    }
}
