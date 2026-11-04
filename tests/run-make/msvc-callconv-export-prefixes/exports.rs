#![feature(abi_vectorcall)]
#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

#[unsafe(export_name = "\u{1}ctl_vectorcall")]
pub extern "vectorcall" fn escaped(_: u32, _: u32) {}

#[unsafe(export_name = "?q_vectorcall")]
pub extern "vectorcall" fn question(_: u32, _: u32) {}

#[unsafe(export_name = "normal_vectorcall")]
pub extern "vectorcall" fn normal(_: u32, _: u32) {}
