#![crate_type = "staticlib"]

fn internal_multiply(a: i32, b: i32) -> i32 {
    a * b
}

#[no_mangle]
pub extern "C" fn libb_multiply(a: i32, b: i32) -> i32 {
    internal_multiply(a, b)
}

#[no_mangle]
pub extern "C" fn libb_greet() -> i32 {
    99
}
