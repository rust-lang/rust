#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
// Basic Wasm Driver Example

mod sys {
    #[link(wasm_import_module = "thing.sys")]
    extern "C" {
        pub fn log(ptr: *const u8, len: u32, level: u32) -> i32;
        pub fn mmio_read32(handle: u32, offset: u32) -> i32;
        pub fn mmio_write32(handle: u32, offset: u32, value: u32) -> i32;
        pub fn sleep_ms(ms: u32) -> i32;
    }
}

#[no_mangle]
pub extern "C" fn init(device_handle: u32) -> i32 {
    let msg = "Hello from Wasm Driver!";
    unsafe {
        sys::log(msg.as_ptr(), msg.len() as u32, 4); // Level 4 = Debug

        // Test MMIO: Write 0xDEADBEEF to offset 0x10, read it back
        let offset = 0x10;
        sys::mmio_write32(device_handle, offset, 0xDEADBEEFu32);

        let val = sys::mmio_read32(device_handle, offset);

        if val == 0xDEADBEEFu32 as i32 {
            let success_msg = "MMIO Readback verified: DEADBEEF";
            sys::log(success_msg.as_ptr(), success_msg.len() as u32, 4);
        } else {
            let fail_msg = "MMIO Readback failed!";
            sys::log(fail_msg.as_ptr(), fail_msg.len() as u32, 1);
            return -1;
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn tick(_device_handle: u32, now_ms: u64) -> i32 {
    unsafe {
        // Just log a tick every now and then
        if now_ms % 1000 < 50 {
            let msg = "Tick...";
            sys::log(msg.as_ptr(), msg.len() as u32, 4);
        }
    }
    0
}
