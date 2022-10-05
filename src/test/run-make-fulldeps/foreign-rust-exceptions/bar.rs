#![crate_type = "cdylib"]
#![feature(c_unwind)]

#[no_mangle]
extern "C-unwind" fn panic() {
    panic!();
}
