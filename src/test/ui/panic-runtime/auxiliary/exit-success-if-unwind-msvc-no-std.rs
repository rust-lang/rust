// compile-flags:-C panic=unwind
// no-prefer-dynamic

#![no_std]
#![crate_type = "rlib"]
#![feature(core_intrinsics)]

struct Bomb;

impl Drop for Bomb {
    fn drop(&mut self) {
        #[link(name = "kernel32")]
        extern "system" {
            fn ExitProcess(code: u32) -> !;
        }
        unsafe {
            ExitProcess(0);
        }
    }
}

pub fn bar(f: fn()) {
    let _bomb = Bomb;
    f();
}

use core::panic::PanicInfo;

#[panic_handler]
fn handle_panic(_: &PanicInfo) -> ! {
    core::intrinsics::abort();
}
