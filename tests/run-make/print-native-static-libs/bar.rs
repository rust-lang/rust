#[no_mangle]
pub extern "C" fn my_bar_add(left: i32, right: i32) -> i32 {
    // Obviously makes no sense but...
    unsafe {
        g_free(std::ptr::null_mut());
        g_free2(std::ptr::null_mut());
    }
    left + right
}

#[link(name = "glib-2.0")]
extern "C" {
    fn g_free(p: *mut ());
}

#[link(name = "glib-2.0")]
extern "C" {
    fn g_free2(p: *mut ());
}
