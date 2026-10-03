extern "C" {
    fn my_function();
}

#[no_mangle]
pub extern "C" fn rust_entry() {
    unsafe {
        my_function();
    }
}
