#![crate_type = "staticlib"]

mod internal {
    pub fn compute(v: i32) -> i32 {
        v * 3 + 1
    }
}

#[no_mangle]
pub extern "C" fn liba_process(v: i32) -> i32 {
    internal::compute(v)
}

#[no_mangle]
pub extern "C" fn liba_answer() -> i32 {
    42
}
