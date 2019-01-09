fn main() {}

#[allow(clippy::result_unwrap_used)]
fn temporary_cstring() {
    use std::ffi::CString;

    CString::new("foo").unwrap().as_ptr();
}
