#![crate_type = "cdylib"]

#[no_mangle]
extern "C-unwind" fn panic() {
    panic!();
}
