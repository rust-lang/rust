#[no_mangle]
pub extern "C" fn add_numbers(a: i32, b: i32) -> i32 {
    private_add(a, b)
}

fn private_add(a: i32, b: i32) -> i32 {
    a + b
}

#[no_mangle]
pub extern "C" fn get_message() -> *const u8 {
    b"Hello from cdylib\0".as_ptr()
}
