// build-pass
// compile-flags:-C panic=abort
// aux-build:exit-success-if-unwind-msvc-no-std.rs
// no-prefer-dynamic
// only-msvc
// We don't run this executable because it will hang in `rust_begin_unwind`

#![no_std]
#![no_main]
#![windows_subsystem = "console"]
#![feature(panic_abort)]

extern crate exit_success_if_unwind_msvc_no_std;
extern crate panic_abort;

use core::panic::PanicInfo;

#[panic_handler]
fn handle_panic(_: &PanicInfo) -> ! {
    loop {}
}

#[no_mangle]
pub unsafe extern "C" fn memcpy(dest: *mut u8, _src: *const u8, _n: usize) -> *mut u8 {
    dest
}

#[no_mangle]
pub unsafe extern "C" fn memmove(dest: *mut u8, _src: *const u8, _n: usize) -> *mut u8 {
    dest
}

#[no_mangle]
pub unsafe extern "C" fn memset(mem: *mut u8, _val: i32, _n: usize) -> *mut u8 {
    mem
}

#[no_mangle]
pub unsafe extern "C" fn memcmp(_mem1: *const u8, _mem2: *const u8, _n: usize) -> i32 {
    0
}

#[no_mangle]
#[used]
static _fltused: i32 = 0;

#[no_mangle]
pub extern "C" fn mainCRTStartup() {
    exit_success_if_unwind_msvc_no_std::bar(main);
}

fn main() {
    panic!();
}
