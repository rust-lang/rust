extern crate bar;

#[no_mangle]
pub extern "C" fn my_foo_add(left: i32, right: i32) -> i32 {
    // Obviously makes no sense but...
    unsafe {
        init(std::ptr::null_mut());
    }
    bar::my_bar_add(left, right)
}

#[link(name = "systemd")]
extern "C" {
    fn init(p: *mut ());
}
