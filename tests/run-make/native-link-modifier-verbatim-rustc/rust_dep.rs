extern "C" {
    fn upstream_native_f() -> i32;
}

pub fn rust_dep() {
    unsafe {
        assert!(upstream_native_f() == 0);
    }
}
