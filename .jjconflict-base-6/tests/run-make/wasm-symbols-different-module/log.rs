//! Issue #56309

#![crate_type = "cdylib"]

#[link(wasm_import_module = "test")]
extern "C" {
    fn log(message_data: u32, message_size: u32);
}

#[no_mangle]
pub fn main() {
    let message = "Hello, world!";
    unsafe {
        log(message.as_ptr() as u32, message.len() as u32);
    }
}
