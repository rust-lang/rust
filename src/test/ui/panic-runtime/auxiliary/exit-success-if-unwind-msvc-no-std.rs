// compile-flags:-C panic=unwind
// no-prefer-dynamic

#![no_std]
#![crate_type = "rlib"]

struct Bomb;

impl Drop for Bomb {
    fn drop(&mut self) {
        #[link(name = "kernel32")]
        extern "C" {
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
