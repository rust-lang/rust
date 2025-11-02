#![crate_type = "cdylib"]

pub extern "C" fn add(a: u64, b: u64) -> u64 {
    a + b
}
