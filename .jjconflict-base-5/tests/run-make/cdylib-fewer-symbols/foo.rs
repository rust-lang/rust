#![crate_type = "cdylib"]

#[no_mangle]
pub extern "C" fn foo() -> u32 {
    3
}
