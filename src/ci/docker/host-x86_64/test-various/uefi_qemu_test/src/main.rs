// Code is adapted from this hello world example:
// https://doc.rust-lang.org/nightly/rustc/platform-support/unknown-uefi.html

#![no_main]
#![no_std]

use core::{panic, ptr};

use r_efi::efi::{Char16, Handle, RESET_SHUTDOWN, Status, SystemTable};

#[panic_handler]
fn panic_handler(_info: &panic::PanicInfo) -> ! {
    loop {}
}

#[export_name = "efi_main"]
pub extern "C" fn main(_h: Handle, st: *mut SystemTable) -> Status {
    let s = [
        0x0048u16, 0x0065u16, 0x006cu16, 0x006cu16, 0x006fu16, // "Hello"
        0x0020u16, //                                             " "
        0x0057u16, 0x006fu16, 0x0072u16, 0x006cu16, 0x0064u16, // "World"
        0x0021u16, //                                             "!"
        0x000au16, //                                             "\n"
        0x0000u16, //                                             NUL
    ];

    // Print "Hello World!".
    let r = unsafe { ((*(*st).con_out).output_string)((*st).con_out, s.as_ptr() as *mut Char16) };
    if r.is_error() {
        return r;
    }

    // Shut down.
    unsafe {
        ((*((*st).runtime_services)).reset_system)(
            RESET_SHUTDOWN,
            Status::SUCCESS,
            0,
            ptr::null_mut(),
        );
    }

    // This should never be reached because `reset_system` should never
    // return, so fail with an error if we get here.
    Status::UNSUPPORTED
}
