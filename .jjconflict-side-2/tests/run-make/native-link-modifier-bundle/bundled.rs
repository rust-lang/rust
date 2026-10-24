#[link(name = "native-staticlib", kind = "static", modifiers = "+bundle")]
extern "C" {
    pub fn native_func();
}

#[no_mangle]
pub extern "C" fn wrapped_func() {
    unsafe {
        native_func();
    }
}
