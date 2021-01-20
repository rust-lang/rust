#[no_mangle]
pub extern fn args_check() {
    assert_ne!(std::env::args_os().count(), 0);
}
