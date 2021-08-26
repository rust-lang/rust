// compile-flags: -Cmetadata=aux

#![doc(html_root_url = "http://example.com/")]
#![feature(lang_items)]
#![no_std]

#[cfg(windows)]
#[link(name = "vcruntime")]
extern {}

#[cfg(windows)]
#[link(name = "ucrt")]
extern {}

#[cfg(windows)]
#[no_mangle]
#[used]
static _fltused: i32 = 0;
#[cfg(windows)]
#[no_mangle]
#[used]
static __aullrem: i32 = 0;
#[cfg(windows)]
#[no_mangle]
#[used]
static __aulldiv: i32 = 0;

#[cfg(windows)]
#[no_mangle]
extern "system" fn _DllMainCRTStartup(_: *const u8, _: u32, _: *const u8) -> u32 { 1 }

#[lang = "eh_personality"]
fn foo() {}

#[panic_handler]
fn bar(_: &core::panic::PanicInfo) -> ! { loop {} }

/// dox
#[doc(primitive = "pointer")]
pub mod ptr {}
