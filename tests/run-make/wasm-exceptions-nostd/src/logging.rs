extern "C" {
    fn __log_utf8(ptr: *const u8, size: usize);
}

pub fn log_str(text: &str) {
    unsafe {
        __log_utf8(text.as_ptr(), text.len());
    }
}
